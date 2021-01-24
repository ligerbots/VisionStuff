Cython for BGR2HSV plus inRange in a single pass

This module does the equivalent of:
     cv2.cvtColor( ,BGR2HSV, )
     cv2.inRange()
in a single pass.

There are 2 versions:
* bgrtohsv_inrange( image, low_limit, high_limit, output )
  - image is the input image (in BGR)
  - low_limit and high_limit are the arguments for inRange() as Numpy arrays
  - output is the output array. It must be pre-allocated.
  
  This routine will loop through the array and compute whether each pixel passes the threshold. In my testing, it is about 2x faster than the OpenCV routines.

* bgrtohsv_inrange_table( lookup, image, output )
  - lookup is the lookup table (see below)
  - image is the input image
  - output is the output results. It must be pre-allocated.
  
  This routine uses a pre-computed lookup table to know if a pixel passes the thresholding. It is 4-5x faster than the OpenCV routines (in my testing), but there is significant startup cost in computing the lookup table.

* bgrtohsv_inrange_preparetable( low_limit, high_limit )
  - low_limit and high_limit are the arguments for inRange() as Numpy arrays
  - returns the lookup table needed by bgrtohsv_inrange_table
  
  This routine computes the lookup table needed by bgrtohsv_inrange_table. It is moderately costly (about 200 msec on my laptop), so should only be done once and re-used. The lookup table is a 2D array because that was the easy way to re-use the OpenCV routines.

As always, your mileage may vary, so test.
