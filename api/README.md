### Neural Style Transfer
Add styles from famous paintings to any photo in a fraction of a second
## References
[1]: X. Huang and S. Belongie. "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization.", in ICCV, 2017.
[2]: [Original ](https://github.com/xunhuang1995/AdaIN-style)
[3]: [implementation in Tensorflow](https://github.com/lafarinio/adain_for_project)
## How to run a Server
From directory: style_transfer/
1.Install and create virtual environment:
```python
pip install virtualenv
virtualenv venv
#NOTE: You can use any name instead of venv.
```
2.Activate your virtual environment:
```python
source venv\bin\activate
```
or
```python
venv/Scripts/activate
```
3.Setup environment before run the server:
```python
pip install -r requirement.txt
```
4.Run Flask server:
With --port option (default port: 5000):
```python
python .\api\app.py
```
Without --port option :
```python
python .\api\app.py -p port_number
```
5.View API document:
[https://documenter.getpostman.com/view/11282663/SzmmUEAc?version=latest](https://documenter.getpostman.com/view/11282663/SzmmUEAc?version=latest)