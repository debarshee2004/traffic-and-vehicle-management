# Traffic and Vehicle Management

```py
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key=API_KEY)
project = rf.workspace("debarshee2004").project("traffic-and-vehicle-management")
version = project.version(1)
dataset = version.download("yolov9")
```
