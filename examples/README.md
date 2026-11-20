# Examples

`novel.json` is a per-novel sidecar for `gauge import`. Put it inside the novel folder (or next
to a single book file as `<book>.json`):

```
MySerial/
  novel.json
  chapter_001.txt
  chapter_002.txt
```

`library.csv` labels many novels at once from the root of a library folder; `name` is the folder
or file name. Columns are optional; `stars` is your own 1-5 verdict (`user_rating`).

```
Books/
  library.csv
  Ann Author - First Serial/
  second_serial/
  Third Book by E. Pub.epub
```

Every number is optional; a novel with no engagement numbers is still analysed and still
contributes exemplars, it just cannot train the model until you rate it (`gauge rate`).
Full guide: [docs/LOCAL_LIBRARY.md](../docs/LOCAL_LIBRARY.md).
