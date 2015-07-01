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
<tr>
  <td>16 x 16</td>
  <td>0</td>
  <td>10.001</td>
  <td>15.6001</td>
  <td>15.6002</td>
  <td>15.5992</td></tr>
<tr>
  <td>32 x 32</td>
  <td>15.6002</td>
  <td>15.6001</td>
  <td>18.0018</td>
  <td>0.0000</td>
  <td>15.5992</td></tr>
<tr>
  <td>64 x 64</td>
  <td>62.4008</td>
  <td>15.6001</td>
  <td>31.2002</td>
  <td>15.6002</td>
  <td>31.1984</td></tr>
<tr>
  <td>128 x 128</td>
  <td>218.4014</td>
  <td>62.4004</td>
  <td>109.2007</td>
  <td>46.8006</td>
  <td>93.5953</td></tr>
<tr>
  <td>256 x 256</td>
  <td>859.3808</td>
  <td>381.000</td>
  <td>436.8028</td>
  <td>218.4028</td>
  <td>343.1956</td></tr>
<tr>
  <td>512 x 512</td>
  <td>3337.8008</td>
  <td>795.6051</td>
  <td>1716.0220</td>
  <td>797.6104</td>
  <td>1403.982</td></tr>
<tr>
  <td>1024 x 1024</td>
  <td>13910.1903</td
  ><td>3139.1574</td>
  <td>6790.0425</td>
  <td>3224.1553</td>
  <td>5551.929</td></tr>
<tr>
  <td>2048 x 2048</td>
  <td>53565.8000</td>
  <td>12548.4000</td>
  <td>27091.4000</td>
  <td>12542.5608</td>
  <td>22258.725</td></tr>
</table>

