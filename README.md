# mmd_tool
Windows用MMDツールライブラリ

## Command

 1. `conda create -n mmd_tool pip python=3.9`
 2. `conda activate mmd_tool`
 3. `pip install -r requirements.txt`
 4. `<ライブラリパス>\bezier\extra-dll\bezier-2a44d276.dll` を `<ライブラリパス>\bezier\bezier.dll` に置く
 
 
## profile
 
 1. `cls & python -m cProfile -s tottime example.py`



```
color.rgb = ambient;
color.a = diffuse.a;
color *=  colorMap;
shadow = color;
color.rgb += pow(dot(H, N), specularPower) * specular;
color = lerp(shadow, color, max(0.0f, dot(N, L) * 3));
```
