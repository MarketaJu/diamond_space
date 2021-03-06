# Diamond Space package for Python

An implementation of piece-wise linear cascaded Hough Transform from paper [_Real Projective Plane Mapping for Detection of Orthogonal Vanishing Points, BMVC 2013_](http://www.fit.vutbr.cz/research/groups/graph/pclines/pub_page.php?id=2013-BMVC-VanPoints)


```bibtex
@INPROCEEDINGS{
   author = {Mark\'{e}ta Dubsk\'{a} and Adam Herout},
   title = {Real Projective Plane Mapping for Detection of Orthogonal	Vanishing Points},
   pages = {1--10},
   booktitle = {Proceedings of BMVC 2013},
   year = {2013},
   location = {Bristol, GB},
   publisher = {The British Machine Vision Association and Society for Pattern Recognition},   
   language = {english}
}
```

# Installation

Requirements are:
* `numpy`
* `scikit-image`

To install [PyPI](https://pypi.org/project/diamond-space/) package:
```
pip install diamond_space
```

Or download an archive from [Releases](https://github.com/MarketaJu/diamond_space/releases) and install it manually.

# Examples

We have a few examples in Colab notebooks where you can mess around with the package.

* [Synthetic lines intersecting in one point](https://colab.research.google.com/drive/17QerSgbe3mG30Ge18skzo8RNYPv24FbQ?usp=sharing)
* [Point projection from Cartesian coordinate system to Diamond space](https://colab.research.google.com/drive/1pN8ro2GGB13zqvGDDAz_g7Sey0Kvh8Ih?usp=sharing)
* [Vanishing point detection in image](https://colab.research.google.com/drive/1mtMktfw72-bzQlHUff7q2SWFcMbRokb8?usp=sharing) 
