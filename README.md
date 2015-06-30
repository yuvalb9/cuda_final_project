# cuda_final_project
final project of cuda @ MTA.ac.il


Tests results:

All tests were done with 200 epochs.
all measurments are in ms.

GT960
======
board size  || CPU time ||  kernel0  ||  kernel1  ||  kernel2  ||  kernel3  ||
==============================================================================
  16 x 16   ||          ||           ||           ||           ||           ||
  32 x 32   ||          ||           ||           ||           ||           ||
  16 x 64   ||          ||           ||           ||           ||           ||
 128 x 128  ||          ||           ||           ||           ||           ||
 256 x 256  ||          ||           ||           ||           ||           ||
 512 x 512  ||          ||           ||           ||           ||           ||
1024 x 1024 ||  7486    ||    555    ||   2125    ||    538    ||    686    ||
2048 x 2048 ||          ||           ||           ||           ||           ||
4096 x 4096 ||          ||           ||           ||           ||           ||
8192 x 8192 ||          ||           ||           ||           ||           ||


