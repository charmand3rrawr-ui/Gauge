# Examples

`novel.json` is a sidecar for `gauge import`. Put it next to your chapter files:

```
MySerial/
  novel.json
  chapter_001.txt
  chapter_002.txt
  ...
```

Then `gauge import MySerial`. Every number is optional; a novel with no engagement numbers is
still analysed and still contributes exemplars, it just cannot be used to train the model.
