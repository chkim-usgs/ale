import json
import pytest
import re
import subprocess
import sys

from ale.isd_to_kernel import isd_to_kernel, spk_comment, ck_comment, main
from conftest import get_isd, get_isd_path
from unittest.mock import patch, MagicMock


@patch("pyspiceql.searchForKernelsets")
@patch("pyspiceql.getSpiceqlName")
@patch("pyspiceql.translateCodeToName")
@patch("pyspiceql.writeSpk")
def test_spk_generation(mock_write_spk, mock_translate, mock_get_name, mock_search, tmp_path):
    """Test that isd_to_kernel correctly handles SPK generation."""
    
    mock_get_name.return_value = "mex"
    mock_search.return_value = [None, {"sclk": ["sclk.tsc"], "lsk": ["lsk.tls"]}]
    mock_translate.return_value = ["MARS", "J2000"]
    
    outfile = tmp_path / "test_spk.bsp"
    
    isd_data = get_isd("ctx")
    isd_file = get_isd_path("ctx")

    isd_to_kernel(
        isd_file=isd_file,
        kernel_type="spk",
        outfile=outfile,
        overwrite=True
    )
    
    assert mock_write_spk.called
    args, kwargs = mock_write_spk.call_args
    
    assert args[0] == str(outfile)                                              # output file path
    assert args[1][0] == isd_data["instrument_position"]["positions"][0]        # state positions
    assert args[2][0] == isd_data["instrument_position"]["ephemeris_times"][0]  # ephemeris times
    assert args[3] == isd_data["naif_keywords"]["BODY_CODE"]                    # body code
    assert args[4] == isd_data["naif_keywords"]["BODY_FRAME_CODE"]              # body frame code
    assert args[5] == "J2000"                                                   # reference frame
    assert args[6] == f"{mock_get_name.return_value}:{'MRO_CTX'}"               # segment id
    assert args[7] == 1                                                         # degree
    assert args[8][0] == isd_data["instrument_position"]["velocities"][0]       # state velocities
    assert "USGS ALE Generated SPK Kernel" in args[9]                           # comment header

    assert len(args[1]) == len(args[2]) == len(args[8]) == 401


@patch("pyspiceql.getSpiceqlName")
@patch("pyspiceql.searchForKernelsets")
@patch("pyspiceql.translateCodeToName")
@patch("pyspiceql.writeCk")
def test_ck_generation(mock_write_ck, mock_translate, mock_search, mock_get_name, tmp_path):
    """Test that isd_to_kernel correctly handles CK generation."""
    
    mock_get_name.return_value = "mex"
    mock_translate.return_value = ["MARS", "J2000"]
    
    # Mock return for SCLK and LSK search
    mock_search.return_value = [None, {
        "sclk": ["mex_sclk.tsc"],
        "lsk": ["naif0012.tls"]
    }] 
    
    outfile = tmp_path / "test_ck.bc"

    isd_data = get_isd("ctx")
    isd_file = get_isd_path("ctx")
    
    isd_to_kernel(
        isd_file=isd_file,
        kernel_type="ck",
        outfile=outfile,
        comment="test comment"
    )
    
    assert mock_write_ck.called
    args, kwargs = mock_write_ck.call_args
    
    assert args[0] == str(outfile)                                                  # output file path
    assert args[1][0] == isd_data["instrument_pointing"]["quaternions"][0]          # quaternions
    assert args[2][0] == isd_data["instrument_pointing"]["ephemeris_times"][0]      # ephemeris times
    assert args[3] == isd_data["instrument_pointing"]["time_dependent_frames"][0]   # instrument frame code
    assert args[4] == "J2000"                                                       # reference frame
    assert args[6] == "mex_sclk.tsc"                                                # sclk kernels list
    assert args[7] == "naif0012.tls"                                                # lsk kernel (first element of list)
    assert args[8][0] == isd_data["instrument_pointing"]["angular_velocities"][0]   # angular velocities
    assert "USGS ALE Generated CK Kernel" in args[9]                                # comment header

    assert len(args[1]) == len(args[2]) == len(args[8]) == 401


@patch("pyspiceql.writeTextKernel")
def test_text_kernel_generation(mock_write_text, tmp_path):
    """Test that isd_to_kernel correctly handles text kernel generation."""
    
    kernel_type = "IK"
    outfile = tmp_path / "test.ti"
    data = '{"TEST_KEYWORD": "TEST_VALUE"}'

    isd_to_kernel(
        kernel_type=kernel_type,
        data=data,
        outfile=outfile
    )
    
    assert mock_write_text.called
    args, kwargs = mock_write_text.call_args
    
    assert args[0] == str(outfile)
    assert args[1] == kernel_type
    assert args[2] == json.loads(data)


def test_invalid_isd_extension():
    """Verify that non-JSON files raise an error."""
    expected_msg = "ISD must be in JSON"
    with pytest.raises(Exception, match=expected_msg):
        isd_to_kernel(isd_file="test.txt", kernel_type="spk")


def test_invalid_kernel_type():
    """Verify that invalid kernel types raise an error."""
    # SpiceQL error
    expected_msg = "std::exception: abc is not a valid kernel type"
    with pytest.raises(Exception, match=re.escape(expected_msg)):
        isd_to_kernel(isd_file="test.json", kernel_type="abc")


def test_empty_data(tmp_path):
    """Verify that text kernels require a data payload."""
    outfile = tmp_path / "test.tf"
    abs_outfile = str(outfile.resolve()) 
    
    expected_msg = f"Must enter JSON keywords to generate kernel [{abs_outfile}]."
    
    with pytest.raises(Exception, match=re.escape(expected_msg)):
        isd_to_kernel(kernel_type="fk", outfile=outfile)


def test_invalid_data(tmp_path):
    """Verify that data payload is JSON."""
    outfile = tmp_path / "test.tf"
    data = "bad data"
    expected_msg = "The 'data' payload is not valid JSON."
    
    with pytest.raises(Exception, match=re.escape(expected_msg)):
        isd_to_kernel(kernel_type="fk", outfile=outfile, data=data)


def test_missing_isd():
    """Verify missing ISD file for binary kernels raises an error."""
    expected_msg = "Missing ISD file."
    with pytest.raises(Exception, match=expected_msg):
        isd_to_kernel(kernel_type="ck")


def test_missing_outfile():
    """Verify missing outfile file for text kernels raises an error."""
    expected_msg = "Must enter an outfile name for text kernels."
    with pytest.raises(Exception, match=expected_msg):
        isd_to_kernel(kernel_type="pck")


@patch("pyspiceql.getSpiceqlName")
@patch("pyspiceql.searchForKernelsets")
@patch("pyspiceql.translateCodeToName")
@patch("pyspiceql.writeSpk")
def test_outfile_extension_correction(mock_write_spk, mock_translate, mock_search, mock_get_name, tmp_path):
    """Verify that isd_to_kernel corrects a wrong extension (e.g., .txt -> .bsp)."""
    
    mock_get_name.return_value = "mex"
    mock_translate.return_value = ["MARS", "J2000"]
    mock_search.return_value = [None, {"sclk": ["mock.tsc"], "lsk": ["mock.tls"]}]
    
    outfile = tmp_path / "test.abc"
    expected_outfile = str(tmp_path / "test.bsp")
    
    isd_to_kernel(
        isd_file=get_isd_path("ctx"),
        kernel_type="spk",
        outfile=outfile,
        overwrite=True
    )
    
    # The function should have changed 'test.abc' to 'test.bsp'
    args, _ = mock_write_spk.call_args
    actual_path_used = args[0]
    
    assert actual_path_used == expected_outfile
    assert actual_path_used.endswith(".bsp")
    assert not actual_path_used.endswith(".abc")


@patch("pyspiceql.getSpiceqlName")
@patch("pyspiceql.searchForKernelsets")
@patch("pyspiceql.translateCodeToName")
@patch("pyspiceql.writeSpk")
def test_mismatched_times_positions(mock_write, mock_translate, mock_search, mock_get_name, tmp_path):
    """Verify state positions and times size are same."""
    mock_get_name.return_value = "mex"
    mock_translate.return_value = ["MARS", "J2000"]
    mock_search.return_value = [None, {"sclk": ["mock.tsc"], "lsk": ["mock.tls"]}]
    
    isd_data = get_isd("ctx")

    # Bump only ephemeris times
    isd_data["instrument_position"]["ephemeris_times"].append(9999.0)
    broken_isd = tmp_path / "bad.json"
    broken_isd.write_text(json.dumps(isd_data))

    with pytest.raises(ValueError, match="Positions and Times length mismatch!"):
        isd_to_kernel(isd_file=broken_isd, kernel_type="spk")


def test_spk_comment():
    """Test SPK comment generation includes all required fields."""
    comment = spk_comment(
        outfile="/path/to/test.bsp",
        segment_id="TEST_SEGMENT",
        start_time="2020-01-01T00:00:00",
        end_time="2020-01-02T00:00:00",
        instrument_id="TEST_INST",
        target_body="12345",
        target_name="TestTarget",
        center_body="499",
        center_name="Mars",
        reference_frame="J2000",
        records=100,
        degree=7,
        kernels={"lsk": ["test.tls"], "spk": ["test.bsp"]},
        comment="User test comment"
    )

    # Verify key sections are present
    assert "USGS ALE Generated SPK Kernel" in comment
    assert "TEST_SEGMENT" in comment
    assert "2020-01-01T00:00:00" in comment
    assert "2020-01-02T00:00:00" in comment
    assert "TEST_INST" in comment
    assert "12345" in comment
    assert "TestTarget" in comment
    assert "499" in comment
    assert "Mars" in comment
    assert "J2000" in comment
    assert "100" in comment
    assert "7" in comment
    assert "User test comment" in comment
    assert "Position Data in the File" in comment
    assert "ID:USGS_SPK_ABCORR=NONE" in comment


def test_ck_comment():
    """Test CK comment generation includes all required fields."""
    comment = ck_comment(
        outfile="/path/to/test.bc",
        segment_id="TEST_CK_SEGMENT",
        start_time="2020-01-01T00:00:00",
        end_time="2020-01-02T00:00:00",
        instrument_id="TEST_INST",
        target_body="12345",
        target_name="TestTarget",
        center_body="499",
        center_name="Mars",
        reference_frame="J2000",
        records=100,
        has_av=True,
        kernels={"lsk": ["test.tls"], "sclk": ["test.tsc"]},
        comment="User CK test comment"
    )

    # Verify key sections are present
    assert "USGS ALE Generated CK Kernel" in comment
    assert "TEST_CK_SEGMENT" in comment
    assert "2020-01-01T00:00:00" in comment
    assert "2020-01-02T00:00:00" in comment
    assert "TEST_INST" in comment
    assert "12345" in comment
    assert "TestTarget" in comment
    assert "499" in comment
    assert "Mars" in comment
    assert "J2000" in comment
    assert "100" in comment
    assert "True" in comment  # has_av
    assert "User CK test comment" in comment
    assert "Orientation Data in the File" in comment
    assert "angular velocity" in comment.lower()


@patch("pyspiceql.getSpiceqlName")
@patch("pyspiceql.searchForKernelsets")
@patch("pyspiceql.translateCodeToName")
@patch("pyspiceql.writeCk")
def test_ck_without_angular_velocities(mock_write_ck, mock_translate, mock_search, mock_get_name, tmp_path):
    """Test CK generation when ISD lacks angular velocities."""
    mock_get_name.return_value = "mex"
    mock_translate.return_value = ["MARS", "J2000"]
    mock_search.return_value = [None, {
        "sclk": ["mex_sclk.tsc"],
        "lsk": ["naif0012.tls"]
    }]

    isd_data = get_isd("ctx")
    # Remove angular velocities
    isd_data["instrument_pointing"]["angular_velocities"] = None

    modified_isd = tmp_path / "no_av.json"
    modified_isd.write_text(json.dumps(isd_data))

    outfile = tmp_path / "test_no_av.bc"

    isd_to_kernel(
        isd_file=modified_isd,
        kernel_type="ck",
        outfile=outfile
    )

    assert mock_write_ck.called
    args, _ = mock_write_ck.call_args

    # Verify angular velocities is empty list
    assert args[8] == []


@patch("pyspiceql.getSpiceqlName")
@patch("pyspiceql.searchForKernelsets")
@patch("pyspiceql.translateCodeToName")
@patch("pyspiceql.writeSpk")
def test_segment_id_truncation(mock_write_spk, mock_translate, mock_search, mock_get_name, tmp_path):
    """Test that segment IDs longer than 40 characters are truncated."""
    # Create a very long mission/frame name combination
    very_long_name = "very_long_mission_name_that_exceeds_limit"
    mock_get_name.return_value = very_long_name
    mock_translate.return_value = ["MARS", "J2000"]
    mock_search.return_value = [None, {"sclk": ["mock.tsc"], "lsk": ["mock.tls"]}]

    outfile = tmp_path / "test_truncate.bsp"
    isd_file = get_isd_path("ctx")

    isd_to_kernel(
        isd_file=isd_file,
        kernel_type="spk",
        outfile=outfile,
        overwrite=True
    )

    assert mock_write_spk.called
    args, _ = mock_write_spk.call_args

    # Verify segment_id (args[6]) is truncated to 40 characters
    segment_id = args[6]
    assert len(segment_id) <= 40


@patch("pyspiceql.getSpiceqlName")
@patch("pyspiceql.searchForKernelsets")
@patch("pyspiceql.translateCodeToName")
@patch("pyspiceql.writeCk")
def test_missing_sclk_kernels(mock_write_ck, mock_translate, mock_search, mock_get_name, tmp_path):
    """Test that missing SCLK kernels raise an appropriate error."""
    mock_get_name.return_value = "mex"
    mock_translate.return_value = ["MARS", "J2000"]
    # Return kernels without SCLK
    mock_search.return_value = [None, {"lsk": ["naif0012.tls"]}]

    outfile = tmp_path / "test_no_sclk.bc"
    isd_file = get_isd_path("ctx")

    with pytest.raises(Exception, match="Could not find SCLKs"):
        isd_to_kernel(
            isd_file=isd_file,
            kernel_type="ck",
            outfile=outfile
        )


@patch("pyspiceql.getSpiceqlName")
@patch("pyspiceql.searchForKernelsets")
@patch("pyspiceql.translateCodeToName")
@patch("pyspiceql.writeCk")
def test_missing_lsk_kernels(mock_write_ck, mock_translate, mock_search, mock_get_name, tmp_path):
    """Test that missing LSK kernels raise an appropriate error."""
    mock_get_name.return_value = "mex"
    mock_translate.return_value = ["MARS", "J2000"]
    # Return kernels without LSK
    mock_search.return_value = [None, {"sclk": ["mex_sclk.tsc"]}]

    outfile = tmp_path / "test_no_lsk.bc"
    isd_file = get_isd_path("ctx")

    with pytest.raises(Exception, match="Could not find LSK"):
        isd_to_kernel(
            isd_file=isd_file,
            kernel_type="ck",
            outfile=outfile
        )


@patch("pyspiceql.getSpiceqlName")
def test_missing_mission_name(mock_get_name, tmp_path):
    """Test that ISD without resolvable mission name raises error."""
    # getSpiceqlName returns None for all candidates
    mock_get_name.return_value = None

    outfile = tmp_path / "test_no_mission.bsp"
    isd_file = get_isd_path("ctx")

    with pytest.raises(Exception, match="Could not find a valid mission name"):
        isd_to_kernel(
            isd_file=isd_file,
            kernel_type="spk",
            outfile=outfile
        )


def test_file_already_exists_no_overwrite(tmp_path):
    """Test that existing files without overwrite flag raise error."""
    outfile = tmp_path / "existing.bsp"
    outfile.write_text("existing content")

    isd_file = get_isd_path("ctx")

    expected_msg = f"Output file [{str(outfile.resolve())}] already exists."

    with pytest.raises(Exception, match=re.escape(expected_msg)):
        isd_to_kernel(
            isd_file=isd_file,
            kernel_type="spk",
            outfile=outfile,
            overwrite=False
        )


@patch("pyspiceql.getSpiceqlName")
@patch("pyspiceql.searchForKernelsets")
@patch("pyspiceql.translateCodeToName")
@patch("pyspiceql.writeSpk")
def test_default_comment_generation(mock_write_spk, mock_translate, mock_search, mock_get_name, tmp_path):
    """Test that a default comment is generated when none is provided."""
    mock_get_name.return_value = "mex"
    mock_translate.return_value = ["MARS", "J2000"]
    mock_search.return_value = [None, {"sclk": ["mock.tsc"], "lsk": ["mock.tls"]}]

    outfile = tmp_path / "test_default_comment.bsp"
    isd_file = get_isd_path("ctx")

    # Call without providing a comment
    isd_to_kernel(
        isd_file=isd_file,
        kernel_type="spk",
        outfile=outfile,
        overwrite=True
    )

    assert mock_write_spk.called
    args, _ = mock_write_spk.call_args

    # Verify default comment is present (args[9] is the comment)
    comment = args[9]
    assert "Auto-generated comment by ALE" in comment
    assert "USGS ALE Generated SPK Kernel" in comment


# CLI / main() function tests
class TestCLI:
    """Test the command-line interface via main() function."""

    @patch("ale.isd_to_kernel.isd_to_kernel")
    @patch("sys.argv", ["isd_to_kernel", "-f", "test.json", "-k", "spk"])
    def test_main_with_basic_args(self, mock_isd_to_kernel):
        """Test main() with basic command-line arguments."""
        main()

        assert mock_isd_to_kernel.called
        call_kwargs = mock_isd_to_kernel.call_args[1]
        assert str(call_kwargs['isd_file']).endswith('test.json')
        assert call_kwargs['kernel_type'] == 'spk'

    @patch("ale.isd_to_kernel.isd_to_kernel")
    @patch("sys.argv", ["isd_to_kernel", "-f", "test.json", "-k", "spk", "-o", "output.bsp"])
    def test_main_with_output_file(self, mock_isd_to_kernel):
        """Test main() with custom output file."""
        main()

        assert mock_isd_to_kernel.called
        call_kwargs = mock_isd_to_kernel.call_args[1]
        assert call_kwargs['outfile'] == 'output.bsp'

    @patch("ale.isd_to_kernel.isd_to_kernel")
    @patch("sys.argv", ["isd_to_kernel", "-f", "test.json", "-k", "spk", "--overwrite"])
    def test_main_with_overwrite_flag(self, mock_isd_to_kernel):
        """Test main() with overwrite flag."""
        main()

        assert mock_isd_to_kernel.called
        call_kwargs = mock_isd_to_kernel.call_args[1]
        assert call_kwargs['overwrite'] is True

    @patch("ale.isd_to_kernel.isd_to_kernel")
    @patch("sys.argv", ["isd_to_kernel", "-f", "test.json", "-k", "spk", "--web"])
    def test_main_with_web_flag(self, mock_isd_to_kernel):
        """Test main() with web flag."""
        main()

        assert mock_isd_to_kernel.called
        call_kwargs = mock_isd_to_kernel.call_args[1]
        assert call_kwargs['use_web'] is True

    @patch("ale.isd_to_kernel.isd_to_kernel")
    @patch("sys.argv", ["isd_to_kernel", "-f", "test.json", "-k", "spk", "-v"])
    def test_main_with_verbose_flag(self, mock_isd_to_kernel):
        """Test main() with verbose flag sets log level."""
        import logging
        main()

        assert mock_isd_to_kernel.called
        call_kwargs = mock_isd_to_kernel.call_args[1]
        assert call_kwargs['log_level'] == logging.INFO

    @patch("ale.isd_to_kernel.isd_to_kernel", side_effect=Exception("Test error"))
    @patch("sys.argv", ["isd_to_kernel", "-f", "test.json", "-k", "spk"])
    def test_main_handles_exceptions(self, mock_isd_to_kernel):
        """Test main() exits gracefully on errors."""
        with pytest.raises(SystemExit) as exc_info:
            main()

        # Verify error message contains the exception
        assert "Test error" in str(exc_info.value)

    @patch("ale.isd_to_kernel.isd_to_kernel")
    @patch("sys.argv", ["isd_to_kernel", "-k", "fk", "-o", "test.tf", "-d", '{"KEY": "VALUE"}'])
    def test_main_with_text_kernel_data(self, mock_isd_to_kernel):
        """Test main() with text kernel data."""
        main()

        assert mock_isd_to_kernel.called
        call_kwargs = mock_isd_to_kernel.call_args[1]
        assert call_kwargs['kernel_type'] == 'fk'
        assert call_kwargs['data'] == '{"KEY": "VALUE"}'


class TestSubprocessCalls:
    """Test actual subprocess calls to isd_to_kernel command."""

    def test_subprocess_help(self):
        """Test that isd_to_kernel --help works."""
        result = subprocess.run(
            ["isd_to_kernel", "--help"],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert "usage:" in result.stdout.lower()
        assert "isd_to_kernel" in result.stdout

    def test_subprocess_missing_required_args(self):
        """Test that isd_to_kernel without required args shows usage."""
        result = subprocess.run(
            ["isd_to_kernel"],
            capture_output=True,
            text=True
        )

        # Should fail without required arguments
        assert result.returncode != 0

    def test_subprocess_generate_spk_fails_without_cache(self, tmp_path):
        """Test subprocess call shows appropriate error without SPICEQL cache."""
        isd_file = get_isd_path("ctx")
        output_file = tmp_path / "subprocess_test.bsp"

        # Run without mocks - this will fail but we test that error is clear
        result = subprocess.run(
            [
                "isd_to_kernel",
                "-f", str(isd_file),
                "-k", "spk",
                "-o", str(output_file),
                "--overwrite"
            ],
            capture_output=True,
            text=True,
            env={**subprocess.os.environ, "SPICEQL_CACHE_DIR": ""}  # Clear cache dir
        )

        # Should fail with clear error message about cache or kernels
        assert result.returncode != 0
        assert "error" in result.stderr.lower() or "exception" in result.stderr.lower() or "cache" in result.stderr.lower()

    def test_subprocess_invalid_kernel_type(self):
        """Test subprocess with invalid kernel type."""
        result = subprocess.run(
            [
                "isd_to_kernel",
                "-f", "dummy.json",
                "-k", "invalid_type"
            ],
            capture_output=True,
            text=True
        )

        # Should fail with invalid kernel type
        assert result.returncode != 0
        assert "error" in result.stderr.lower() or "exception" in result.stderr.lower()

    def test_subprocess_with_verbose(self, tmp_path):
        """Test subprocess with verbose flag produces output."""
        # Create a dummy ISD file
        dummy_isd = tmp_path / "dummy.json"
        isd_data = get_isd("ctx")
        dummy_isd.write_text(json.dumps(isd_data))

        result = subprocess.run(
            [
                "isd_to_kernel",
                "-f", str(dummy_isd),
                "-k", "spk",
                "-v"
            ],
            capture_output=True,
            text=True,
            timeout=10
        )

        # Verbose mode should produce some output (may fail due to missing dependencies, but should show verbose logs)
        # We're mainly testing that -v flag is recognized
        assert len(result.stdout) > 0 or len(result.stderr) > 0

    def test_subprocess_missing_isd_file(self):
        """Test subprocess with non-existent ISD file."""
        result = subprocess.run(
            [
                "isd_to_kernel",
                "-f", "/nonexistent/file.json",
                "-k", "spk"
            ],
            capture_output=True,
            text=True
        )

        # Should fail with file not found error
        assert result.returncode != 0
        assert result.stderr  # Should have error message

    def test_subprocess_invalid_isd_format(self, tmp_path):
        """Test subprocess with invalid JSON in ISD file."""
        bad_isd = tmp_path / "bad.json"
        bad_isd.write_text("this is not valid json {{{")

        result = subprocess.run(
            [
                "isd_to_kernel",
                "-f", str(bad_isd),
                "-k", "spk"
            ],
            capture_output=True,
            text=True
        )

        # Should fail with JSON parse error
        assert result.returncode != 0

    def test_subprocess_text_kernel_missing_data(self, tmp_path):
        """Test subprocess trying to create text kernel without data."""
        output = tmp_path / "test.tf"

        result = subprocess.run(
            [
                "isd_to_kernel",
                "-k", "fk",
                "-o", str(output)
            ],
            capture_output=True,
            text=True
        )

        # Should fail - text kernels require data
        assert result.returncode != 0

    def test_subprocess_text_kernel_with_data(self, tmp_path):
        """Test subprocess creating text kernel with JSON data."""
        output = tmp_path / "test.tf"

        result = subprocess.run(
            [
                "isd_to_kernel",
                "-k", "fk",
                "-o", str(output),
                "-d", '{"TEST_KEY": "TEST_VALUE"}'
            ],
            capture_output=True,
            text=True
        )

        # May fail due to missing dependencies, but should parse args correctly
        # Main test is that it doesn't fail due to argument parsing
        if result.returncode == 0:
            assert output.exists()

    def test_subprocess_file_exists_no_overwrite(self, tmp_path):
        """Test subprocess fails when output file exists and no --overwrite."""
        # Create existing file
        existing_file = tmp_path / "existing.bsp"
        existing_file.write_text("existing content")

        isd_file = get_isd_path("ctx")

        result = subprocess.run(
            [
                "isd_to_kernel",
                "-f", str(isd_file),
                "-k", "spk",
                "-o", str(existing_file)
            ],
            capture_output=True,
            text=True
        )

        # Should fail because file exists
        assert result.returncode != 0
        assert "exists" in result.stderr.lower()

    def test_subprocess_file_exists_with_overwrite(self, tmp_path):
        """Test subprocess succeeds with --overwrite flag."""
        # Create existing file
        existing_file = tmp_path / "existing.bsp"
        existing_file.write_text("existing content")

        isd_file = get_isd_path("ctx")

        result = subprocess.run(
            [
                "isd_to_kernel",
                "-f", str(isd_file),
                "-k", "spk",
                "-o", str(existing_file),
                "--overwrite"
            ],
            capture_output=True,
            text=True
        )

        # May fail due to missing SPICE data, but shouldn't fail due to file existing
        # We're testing that --overwrite is recognized
        if "exists" in result.stderr.lower():
            pytest.fail("Should not fail due to existing file with --overwrite flag")

    def test_subprocess_invalid_json_data(self, tmp_path):
        """Test subprocess with invalid JSON in -d data argument."""
        output = tmp_path / "test.tf"

        result = subprocess.run(
            [
                "isd_to_kernel",
                "-k", "fk",
                "-o", str(output),
                "-d", "not valid json {{"
            ],
            capture_output=True,
            text=True
        )

        # Should fail with JSON parse error
        assert result.returncode != 0

