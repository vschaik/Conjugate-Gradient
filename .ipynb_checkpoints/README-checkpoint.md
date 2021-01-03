# An Introduction to the Conjugate Gradient Method Without the Agonizing Pain

The Conjugate Gradient Method is the most prominent iterative method for solving sparse systems of linear equations. Unfortunately, many textbook treatments of the topic are written with neither illustrations nor intuition, and their victims can be found to this day babbling senselessly in the corners of dusty libraries. For this reason, a deep, geometric understanding of the method has been reserved for the elite brilliant few who have painstakingly decoded the mumblings of their forebears. Nevertheless, the Conjugate Gradient Method is a composite of simple, elegant ideas that almost anyone can understand. Of course, a reader as intelligent as yourself will learn them almost effortlessly.
The idea of quadratic forms is introduced and used to derive the methods of Steepest Descent, Conjugate Directions, and Conjugate Gradients. Eigenvectors are explained and used to examine the convergence of the Jacobi Method, Steepest Descent, and Conjugate Gradients. Other topics include preconditioning and the nonlinear Conjugate Gradient Method. I have taken pains to make this article easy to read. Sixty-six illustrations are provided. Dense prose is avoided. Concepts are explained in several different ways. Most equations are coupled with an intuitive interpretation.

By: [Jonathan Richard Shewchuk](https://people.eecs.berkeley.edu/~jrs/) August 1994

Converted to a notebook by [André van Schaik](http://westernsydney.edu.au/bens) in June 2017. You can now zoom images and rotate 3D images, and I've added a few interactive figures with sliders to play with. 

Start reading at [CG00.ipynb](https://github.com/vschaik/Conjugate-Gradient/blob/master/CG00.ipynb)

**Edit** I've updated these in January 2021 to use the %matplotlib widget environment for compatibility with JupyterLab. For this to work you will need to have installed the ipympl package and the jupyterlabs-widget extension. For instructions see [here](https://github.com/matplotlib/ipympl)