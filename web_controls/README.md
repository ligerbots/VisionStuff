# Webcontrols
## Install
Source: https://robotpy.readthedocs.io/projects/pynetworktables2js/en/stable/index.html  
```
pip3 install pynetworktables2js
```
Also set camera settings in webControls/index.js:
```
loadCameraOnConnect({
  host: "IP.OF.SERVER",
  container: '#cam-frame',
  port: CAMPORT,
  image_url: '/stream.mjpg',
  data_url: '/settings.json',
  attrs: {
    width: 0,
    height: 0
  }
});
```
## Start Server
```
cd webControls
python3 -m pynetworktables2js
```
