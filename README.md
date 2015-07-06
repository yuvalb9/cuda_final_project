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
<tr><th>size</th><th>cpu</th><th>kernel0</th><th>kernel1</th><th>kernel2</th><th>kernel3</th><th>kernel4</th><th>kernel5</th><th>kernel6</th></tr>
<tr><td nowrap>16 X 16</td><td nowrap>0</td><td nowrap>0.6001</td><td nowrap>0.6001</td><td nowrap>0.7001</td><td nowrap>0.6001</td><td nowrap>0.7003</td><td nowrap>0.6001</td><td nowrap>0.7001</td></tr>
<tr><td nowrap>32 X 32</td><td nowrap>15.6002</td><td nowrap>11.5011</td><td nowrap>21.7022</td><td nowrap>10.8011</td><td nowrap>13.3013</td><td nowrap>19.6078</td><td nowrap>11.5011</td><td nowrap>14.7015</td></tr>
<tr><td nowrap>64 X 64</td><td nowrap>62.4008</td><td nowrap>21.8022</td><td nowrap>59.806</td><td nowrap>22.5023</td><td nowrap>30.203</td><td nowrap>50.2201</td><td nowrap>21.8022</td><td nowrap>22.3022</td></tr>
<tr><td nowrap>128 X 128</td><td nowrap>218.4014</td><td nowrap>69.4069</td><td nowrap>210.021</td><td nowrap>69.607</td><td nowrap>93.6094</td><td nowrap>175.7703</td><td nowrap>62.3062</td><td nowrap>66.3066</td></tr>
<tr><td nowrap>256 X 256</td><td nowrap>859.3808</td><td nowrap>254.1254</td><td nowrap>802.7803</td><td nowrap>248.6249</td><td nowrap>338.6339</td><td nowrap>631.7526</td><td nowrap>229.0229</td><td nowrap>228.8229</td></tr>
<tr><td nowrap>512 X 512</td><td nowrap>3337.8008</td><td nowrap>1081.1081</td><td nowrap>4166.2166</td><td nowrap>1094.4094</td><td nowrap>1393.6393</td><td nowrap>2656.9624</td><td nowrap>972.1972</td><td nowrap>963.5964</td></tr>
<tr><td nowrap>1024 X 1024</td><td nowrap>13910.1903</td><td nowrap>4313.1313</td><td nowrap>17002.8001</td><td nowrap>4831.2831</td><td nowrap>5547.1547</td><td nowrap>10543.1156</td><td nowrap>3836.2836</td><td nowrap>3944.9945</td></tr>
<tr><td nowrap>2048 X 2048</td><td nowrap>53565.8</td><td nowrap>17166.0164</td><td nowrap>67750.8744</td><td nowrap>18078.3076</td><td nowrap>22701.1332</td><td nowrap>43434.1341</td><td nowrap>15278.2277</td><td nowrap>15430.9429</td></tr>

</table>

