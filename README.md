# cuda_final_project<br>
final project of cuda @ MTA.ac.il<br>
<br>
<br>
Tests results:<br>
<br>
All tests were done with 200 epochs.<br>
all measurments are in ms.<br>
<br>
GT220<br>
<table border="1">
<tr><th>board size  </th><th> CPU time </th><th>  kernel0  </th><th>  kernel1  </th><th> kernel2  </th><th>  kernel3  </th></tr>
<tr><td>16 x 16</td><td></td><td></td><td></td><td></td><td>15.5992</td></tr>
<tr><td>32 x 16</td><td></td><td></td><td></td><td></td><td>15.5992</td></tr>
<tr><td>64 x 16</td><td></td><td></td><td></td><td></td><td>31.1984</td></tr>
<tr><td>128 x 16</td><td></td><td></td><td></td><td></td><td>93.5953</td></tr>
<tr><td>256 x 256</td><td></td><td></td><td></td><td></td><td>343.1956</td></tr>
<tr><td>512 x 512</td><td>3337.8008</td><td>811.0544</td><td></td><td></td><td>1403.982</td></tr>
<tr><td>1024 x 1024</td><td>13910.1903</td><td>3139.1574</td><td>6790.0425</td><td>3224.1553</td><td>5551.929</td></tr>
<tr><td>2048 x 2048</td><td></td><td></td><td></td><td></td><td>22258.725</td></tr>
<tr><td>4096 x 4096</td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td>8192 x 8192</td><td></td><td></td><td></td><td></td><td></td></tr>
</table>

