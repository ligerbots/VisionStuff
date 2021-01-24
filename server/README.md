Install (most of) the requirements:
`pip install -r requirements.txt`

Start server manually:
`python -m visionserver.visionserver2020`

Test a finder:
`python -m finders.ballfinder2020 --output_dir ../test_data/test_output --calib_file data/camera_calibration/c930e_848x480_calib.json ../test_data/test_images/2021/galactic-search-images/*`

Benchmark a bgrtohsv_inrange:
`python -m optimization.bgrtohsv_inrange.test ../test_data/test_images/2020/Feeder_Station_Target_clean.jpg`
