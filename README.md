# rubyrizz-data
Data extraction repo for rubyrizz

- [X] Finish annotation
- [ ] Refine the training
- [ ] Check the accuracy

Libraries: PyTorch, Opencv

Command to run the prediction
```
yolo predict model='runs/detect/train6/weights/best.pt' source='data/test/images' save=True
```

## Rules to contribute
Refer to [CONTRIBUTING.md](CONTRIBUTING.md) to make contributions to the repository. 