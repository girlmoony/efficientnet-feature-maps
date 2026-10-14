# この実測例の再生成

入力：`cat.png`（ユーザー提供）。重み：ローカルの `efficientnet_lite1.pth`。重み自体は同梱していません。

リポジトリのルートから、セットアップ済みPythonで実行：

```powershell
python feature_maps.py --weights "C:\Users\thatb\Downloads\efficientnet_lite1.pth" --image examples/cat.png --model efficientnet-lite1 --channel-order rgb --output runs/reproduced
```

`examples/lite1-rgb` は既に出力があるため、再実行時は `runs/reproduced` など別のディレクトリを指定します。

保存された実行結果：

- 入力240×240、RGB、短辺272へBICUBICリサイズ後に中央切り出し。
- 平均 `[0.498, 0.498, 0.498]`、標準偏差 `[0.502, 0.502, 0.502]`。
- CPU / eval / strict=True。空間構造のある出力158か所を保存。
- 出力ID 283のsoftmax値：`0.44515761733055115`。ラベルは未指定なのでクラス名を推測していません。
- 重みと画像のSHA-256、ライブラリバージョンは `lite1-rgb/metadata.json` に記録。

学習時の前処理やライブラリのバージョンが異なる場合、結果が変わる可能性があります。RGBモデルにBGRを入力すると別の入力になり、同じ結果を期待するものではありません。
