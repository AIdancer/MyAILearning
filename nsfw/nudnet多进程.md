### nudenet批量预测
```python
import os
import sys
import time
from nudenet import NudeDetector

detector = NudeDetector()

def check_images(image_paths):
    global detector
    try:
        results = detector.detect_batch(image_paths)
        return results
    except Exception as e:
        return {"error" : str(e)}


if __name__ == '__main__':
    image_handles = []
    process_name = sys.argv[1]
    root_path = sys.argv[2]
    for root, dirs, files in os.walk(root_path):
        for file in files:
            image_handles.append(os.path.join(root, file))

    start = time.time()

    image_handles = image_handles * 10
    n = len(image_handles)
    print(n)
    results = check_images(image_handles)

    end = time.time()

    print(f'process_name:{process_name}    time:{end-start}')
```

### powershell多进程
```ps1

Start-Process -FilePath "python" `
    -ArgumentList "D:\code\model-acc\single-acc.py","p1","D:\data\pic\p1" `
    -RedirectStandardOutput 'o1.txt'

Start-Process -FilePath "python" `
    -ArgumentList "D:\code\model-acc\single-acc.py","p2","D:\data\pic\p2" `
    -RedirectStandardOutput 'o2.txt'

Start-Process -FilePath "python" `
    -ArgumentList "D:\code\model-acc\single-acc.py","p3","D:\data\pic\p3" `
    -RedirectStandardOutput 'o3.txt'

Start-Process -FilePath "python" `
    -ArgumentList "D:\code\model-acc\single-acc.py","p4","D:\data\pic\p4" `
    -RedirectStandardOutput 'o4.txt'
```
