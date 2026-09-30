from pathlib import Path

import pytest

import aioftp


def encode_whole(data):
    encoder = aioftp.NetASCIIEncoder()
    return encoder.transform(data) + encoder.flush()


def encode_bytewise(data):
    encoder = aioftp.NetASCIIEncoder()
    out = bytearray()
    for i in range(len(data)):
        out += encoder.transform(data[i : i + 1])
    out += encoder.flush()
    return bytes(out)


def decode_whole(data):
    decoder = aioftp.NetASCIIDecoder()
    return decoder.transform(data) + decoder.flush()


def decode_bytewise(data):
    decoder = aioftp.NetASCIIDecoder()
    out = bytearray()
    for i in range(len(data)):
        out += decoder.transform(data[i : i + 1])
    out += decoder.flush()
    return bytes(out)


# (netascii on the wire, expected local representation)
DECODE_CASES = [
    (b"", b""),
    (b"plain", b"plain"),
    (b"\n", b"\n"),
    (b"\r\n", b"\n"),
    (b"\r\x00", b"\r"),
    (b"\r", b"\r"),
    (b"a\r\nb", b"a\nb"),
    (b"a\r\x00b", b"a\rb"),
    (b"a\rb", b"a\rb"),
    (b"ab\r", b"ab\r"),
    (b"\r\r\n", b"\r\n"),
    (b"\r\x00\r\n", b"\r\n"),
    (b"x\r\x00\r\ny\r", b"x\r\ny\r"),
    (b"\r\r\r", b"\r\r\r"),
    (b"\x00\xff\r\n\xfe", b"\x00\xff\n\xfe"),
]

LOCAL_CASES = [
    b"",
    b"plain",
    b"\n",
    b"\r",
    b"\r\n",
    b"\r\n\r\n",
    b"a\nb\rc\r\nd\r",
    b"trailing\r",
    b"\x00\x01\xff\n\r",
    bytes(range(256)),
]

WELL_FORMED_NETASCII = [
    b"",
    b"plain",
    b"\r\n",
    b"\r\x00",
    b"a\r\nb\r\x00c",
    b"line1\r\nline2\r\n",
]


@pytest.mark.parametrize("net, local", DECODE_CASES)
def test_netascii_decoder_cases(net, local):
    assert decode_whole(net) == local
    assert decode_bytewise(net) == local


@pytest.mark.parametrize("net, local", DECODE_CASES)
def test_netascii_decoder_all_two_chunk_splits(net, local):
    for i in range(len(net) + 1):
        decoder = aioftp.NetASCIIDecoder()
        result = decoder.transform(net[:i]) + decoder.transform(net[i:]) + decoder.flush()
        assert result == local, f"split at {i}"


@pytest.mark.parametrize("data", LOCAL_CASES)
def test_netascii_encoder_bytewise_matches_whole(data):
    assert encode_bytewise(data) == encode_whole(data)


@pytest.mark.parametrize("data", LOCAL_CASES)
def test_netascii_local_round_trip(data):
    assert decode_whole(encode_whole(data)) == data
    assert decode_bytewise(encode_bytewise(data)) == data


@pytest.mark.parametrize("data", WELL_FORMED_NETASCII)
def test_netascii_wire_round_trip(data):
    assert encode_whole(decode_whole(data)) == data


def test_netascii_decoder_trailing_lone_cr_defined():
    decoder = aioftp.NetASCIIDecoder()
    assert decoder.transform(b"data\r") == b"data"
    assert decoder.flush() == b"\r"
    assert decoder.flush() == b""


def test_netascii_large_stream_not_cached():
    # big input transformed incrementally in odd-sized chunks must match
    # the whole-block result without holding the whole file in memory
    block = bytes(range(256))
    data = block * 4096  # 1 MiB containing every byte value
    whole = decode_whole(data)
    decoder = aioftp.NetASCIIDecoder()
    out = bytearray()
    for i in range(0, len(data), 997):
        out += decoder.transform(data[i : i + 997])
    out += decoder.flush()
    assert bytes(out) == whole
    whole_enc = encode_whole(data)
    encoder = aioftp.NetASCIIEncoder()
    out = bytearray()
    for i in range(0, len(data), 997):
        out += encoder.transform(data[i : i + 997])
    out += encoder.flush()
    assert bytes(out) == whole_enc


LOCAL_TRICKY = b"first\nsecond\rthird\r\nfourth\n\n" b"\x00\xffbinary\r\nlone\r"


@pytest.mark.asyncio
async def test_type_a_upload_download_stream_round_trip(pair_factory):
    async with pair_factory() as pair:
        async with pair.client.upload_stream("foo.txt", conn_type="A") as stream:
            await stream.write(LOCAL_TRICKY)
        # server stored the decoded (local) representation
        async with pair.client.download_stream("foo.txt") as stream:
            assert await stream.read() == LOCAL_TRICKY
        # and TYPE A download transforms it back on the wire
        async with pair.client.download_stream("foo.txt", conn_type="A") as stream:
            assert await stream.read() == LOCAL_TRICKY


@pytest.mark.asyncio
async def test_type_a_server_stor_decodes_bytewise(pair_factory):
    wire = b"a\r\nb\r\x00c\rd\r"
    expect = b"a\nb\rc\rd\r"
    async with pair_factory() as pair:
        async with pair.client.get_stream("STOR foo.txt", "1xx", conn_type="A") as stream:
            for i in range(len(wire)):
                await stream.write(wire[i : i + 1])
        async with pair.client.download_stream("foo.txt") as stream:
            assert await stream.read() == expect


@pytest.mark.asyncio
async def test_type_a_server_retr_encodes(pair_factory):
    local = b"a\nb\rc\r\nd\r"
    wire_expect = encode_whole(local)
    async with pair_factory() as pair:
        async with pair.client.upload_stream("foo.txt") as stream:
            await stream.write(local)
        async with pair.client.get_stream("RETR foo.txt", "1xx", conn_type="A") as stream:
            assert await stream.read() == wire_expect


@pytest.mark.asyncio
async def test_type_a_high_level_upload_download(pair_factory):
    async with pair_factory() as pair:
        async with pair.client.path_io.open(Path("local.txt"), mode="wb") as f:
            await f.write(LOCAL_TRICKY)
        await pair.client.upload("local.txt", conn_type="A")
        await pair.client.download("local.txt", "back", conn_type="A")
        async with pair.client.path_io.open(Path("back/local.txt"), mode="rb") as f:
            assert await f.read() == LOCAL_TRICKY


@pytest.mark.asyncio
async def test_type_a_large_file_round_trip(pair_factory):
    # ~1 MiB, written/read in odd-sized chunks so CR LF pairs are split
    # across blocks; transfer must stream, not cache the whole file
    data = bytes(range(256)) * 4096
    async with pair_factory() as pair:
        async with pair.client.upload_stream("big.bin", conn_type="A") as stream:
            for i in range(0, len(data), 997):
                await stream.write(data[i : i + 997])
        async with pair.client.download_stream("big.bin", conn_type="A") as stream:
            blocks = []
            async for block in stream.iter_by_block(65536):
                blocks.append(block)
            assert b"".join(blocks) == data


@pytest.mark.asyncio
async def test_type_i_binary_untouched(pair_factory):
    data = bytes(range(256)) * 10
    async with pair_factory() as pair:
        async with pair.client.upload_stream("bin", conn_type="I") as stream:
            await stream.write(data)
        async with pair.client.download_stream("bin", conn_type="I") as stream:
            assert await stream.read() == data


@pytest.mark.asyncio
async def test_type_a_rest_refused(pair_factory, expect_codes_in_exception):
    async with pair_factory() as pair:
        await pair.client.command("TYPE A", "200")
        with expect_codes_in_exception("502"):
            await pair.client.command("REST 3", "350")


@pytest.mark.asyncio
async def test_type_a_offset_raises(pair_factory):
    async with pair_factory() as pair:
        await pair.make_server_files("foo.txt", size=1, atom=b"foobar")
        with pytest.raises(ValueError, match="not available for TYPE A"):
            async with pair.client.download_stream("foo.txt", offset=2, conn_type="A"):
                pass
        with pytest.raises(ValueError, match="not available for TYPE A"):
            async with pair.client.upload_stream("bar.txt", offset=2, conn_type="A"):
                pass


@pytest.mark.asyncio
async def test_size_type_i(pair_factory):
    async with pair_factory() as pair:
        await pair.make_server_files("foo.txt", size=1, atom=b"foobar")
        code, info = await pair.client.command("SIZE foo.txt", "213")
        assert info[-1] == "6"


@pytest.mark.asyncio
async def test_size_type_a_refused(pair_factory, expect_codes_in_exception):
    async with pair_factory() as pair:
        await pair.make_server_files("foo.txt", size=1, atom=b"foobar")
        await pair.client.command("TYPE A", "200")
        with expect_codes_in_exception("502"):
            await pair.client.command("SIZE foo.txt", "213")
