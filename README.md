## ___***AnyCharV: Bootstrap Controllable Character Video Generation with Fine-to-Coarse Guidance***___

<div align="center">
<img src='assets/favicon.ico' style="height:100px"></img>




 <a href='https://arxiv.org/abs/2502.08189'><img src='https://img.shields.io/badge/arXiv-2502.08189-b31b1b.svg'></a> &nbsp;
 <a href='https://anycharv.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;
 <a href='https://huggingface.co/harriswen/AnyCharV'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Page-blue'></a>

_[Zhao Wang](https://kyfafyd.wang/)\*, [Hao Wen](https://github.com/wenhao7841)\*, [Lingting Zhu](https://scholar.google.com/citations?user=TPD_P98AAAAJ), [Chengming Shang](), [Yujiu Yang](https://sites.google.com/view/iigroup-thu/about)†, [Qi Dou](https://www.cse.cuhk.edu.hk/~qdou)†_
<br><br>

</div>



## 📝 Changelog
- __[2025.02.17]__: 🔥 Pre-trained model released on huggingface.
- __[2025.02.13]__: Project page is built.



## Details

> Character video generation is a significant real-world application focused on producing high-quality videos featuring specific characters. Recent advancements have introduced various control signals to animate static characters, successfully enhancing control over the generation process. However, these methods often lack flexibility, limiting their applicability and making it challenging for users to synthesize a source character into a desired target scene. To address this issue, we propose a novel framework, ***AnyCharV***, that flexibly generates character videos using arbitrary source characters and target scenes, guided by pose information. Our approach involves a two-stage training process. In the first stage, we develop a base model capable of integrating the source character with the target scene using pose guidance. The second stage further bootstraps controllable generation through a self-boosting mechanism, where we use the generated video in the first stage and replace the fine mask with the coarse one, enabling training outcomes with better preservation of character details. Experimental results demonstrate the effectiveness and robustness of our proposed method.



<div align="center">
    <a href="https://"><img width="1000px" height="auto" src="assets/framework.png"></a>
</div>


## 🛡️ License

This project is under the Apache License 2.0 license. See [LICENSE](LICENSE) for details.

## 📝 Citation

If you find this code useful, please cite in your research papers.
```
@article{wang2025anycharv,
  title={AnyCharV: Bootstrap Controllable Character Video Generation with Fine-to-Coarse Guidance},
  author={Wang, Zhao and Wen, Hao and Zhu, Lingting and Shang, Chengming and Yang, Yujiu and Dou, Qi},
  journal={arXiv preprint arXiv:2502.08189},
  year={2025}
}
```
