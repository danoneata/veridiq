Generate and visualize temporal explanations:
```bash
streamlit veridiq/explanations/show_temporal_explanations.py
```

Generate spatial explanations:
```bash
for c in av-hubert-{a,v} clip fsfm wav2vec videomae; do
    python veridiq/explanations/generate_spatial_explanations.py -c $c
done
```

Visualize spatial explanations:
```bash
streamlit veridiq/explanations/show_spatial_explanations.py
```