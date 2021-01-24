Install (most of) the requirements:
`python3 -m pip install -r requirements.txt`

Start server manually:
`python3 -m visionserver.visionserver2020 --calib_dir data/camera_calibration --test`

Test a finder:
`python3 -m finders.ballfinder2020 --output_dir ../test_data/test_output --calib_file data/camera_calibration/c930e_848x480_calib.json ../test_data/test_images/2021/galactic-search-images/*`

Benchmark a bgrtohsv_inrange:
`python3 -m optimization.bgrtohsv_inrange.test ../test_data/test_images/2020/Feeder_Station_Target_clean.jpg`
