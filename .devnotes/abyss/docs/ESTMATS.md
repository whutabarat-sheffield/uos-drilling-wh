# Estimating Material Thickness

## Packages investigated
 - [pwlf](https://jekel.me/piecewise_linear_fit_py/stubs/pwlf.PiecewiseLinFit.html)
 - [Rupture](https://centre-borelli.github.io/ruptures-docs/)
 - [Jenkspy](https://github.com/mthh/jenkspy)

## Goal

The goal is to estimate the thicknesses of each element in the material stack. In general there are two distinct materials per stack and can be the same. The distinctness refers to the two layers of material being physically separate parts.

## Methodology

The methodology is identifying the places where the material starts and ends from the torque signal. The torque signal tends to change drastically when moving from one material to another due to the changes in properties requiring a different amount of torque. For example, when going between Aluminium and Titanium more torque is required as Titanium is harder.

This type of analysis can be generally referred to as Breakpoint Analysis or identification. A breakpoint is a place where the signal changes distinctly to a new phase or form of behaviour. There are several algorithms which can be used to identify them.

From testing 13 breakpoints was found to work well.

### Piecewise Linear Fitting via Differential Evolution

This is the method used by [BreakpointFit](https://github.com/D-B-Miller/ACSE-EADU_data_analytics/blob/108bf2ab4a98d4eaed3ac0c40227fe2e11e79755/scripts/abyss/src/modelling.py#L1922) in the modelling script. This class seeks to fit a series of straight lines or pieces to the signal with the start and end of each piece being the breakpoints we seek. It uses Differential Evolution with a custom cost function to control the distance between breakpoints so it's not too close or far away from each other. This non-linear constraint is to stop the function from fitting very long lines that are unrealistic and conversely too small.

The main downside with this approach is that it can take several minutes to find the results for a single signal. The upside is it generally achieves accurate results.

### Pruned Exact Linear Time (PELT)

PELT wants to split a signal into a finite number of sections. It prunes them to a manageable set according to a linear cost function. The paper for the method can be found [here](https://www.tandfonline.com/doi/pdf/10.1080/01621459.2012.737745?casa_token=KOGcqO6XqnkAAAAA:5Rqt3txpp2YKWYu9CyIyUvtRNtT6MOY6u6YIVdymZdR8--bwMMAodDfy3MGeqHUyPwCQZkwDbbtCvw). The special bit of PELT is removing breakpoints that can never be a minima of the cost function.The ruptures package supports several cost models; L1, L2 and RBF.

The following plot is from a GUI to adjust the penalty parameter. There are some schemes for choosing penalty value (See BIC or AIC). The slider is for trying different penalty values.

Increasing the penalty value decreases the number of breakpoints and increases fitting time. A near zero penalty results in over 200+ bps for L1, 100+ for L2 fitting with RBF staying below 100. For a penalty value above 4, for this file, the number of breakpoints doesn't change as much.

![pelt-penalty-slider-3](https://user-images.githubusercontent.com/46482002/223149392-71794a8b-0d7b-498f-a41b-72ff117dc738.png)

![pelt-penalty-slider](https://user-images.githubusercontent.com/46482002/223144964-47bafb07-d0b4-4776-91be-b9496ec1e079.png)

![pelt-penalty-slider-2](https://user-images.githubusercontent.com/46482002/223146698-3dce8416-36bf-49c3-acfc-02aa14ca39da.png)

The following plot denotes the number of breakpoints for a certain penalty value. The number of breakpoints stagnates sharply for all three cost models with the L1 model taking the longest to stagnate to the same extent as the others.

![pelt-penalty-nbps](https://user-images.githubusercontent.com/46482002/223157714-7c8f009b-a6ea-4941-91c8-d4bd9c7912f5.png)

The fitting of the breakpoints here is graded visually by how well it fits the changes at approx 14mm, 40mm and 54mm as these are the approx edges of the materials. The cost models RBF appears to perform best due to the number of breakpoints not being as sensitive to the penalty and there being breakpoints close to the stated locations. Sentivity to penalty changes affects the user friendlyness for the non-technical users and for a range of data. It's likely a single penalty value is used across a dataset or likely several datasets so you want a system that's robust enough to work across the variance of values.

### Kernel Changepoint Detection (K-CPD)

With K-CPD, the supported cost models are linear, rbf and cosine. In general, the cosine cost model performed very poorly with either the number of breakpoints not changing above 1 or the breakpoints being towards the ends of the signals.

K-CPD seemed to perform faster in all cases then PELT due to specifying the number of breakpoints when predicting.

This first plot is set for 13 breakpoints and the second is for 30 breakpoints. Incrasing the number of breakpoints seems to primarily focus the new ones along steep slopes towards the actual changepoints. For the Linear kernel, the number of breakpoints is spread over an area with the previous ones being broken up and scattered. For RBF, the previous points are shifted slightly and the ones distributed around it. This has the effect of the new points moving towards the true change points.

![E00401009F45AF14_18080018_ST_2109_27-kernel-models-minsz-3-nbps-13](https://user-images.githubusercontent.com/46482002/223169662-143ecb58-9512-459f-b8e5-8a13c8ba92ad.png)

![E00401009F45AF14_18080018_ST_2109_27-kernel-models-minsz-3-nbps-30](https://user-images.githubusercontent.com/46482002/223169734-e1bb24b0-8d4b-4abc-95a1-e4a2f19bc666.png)

If instead the penalty is used to control the number of breakpoints found, above a value of 2 the distribution of Linear or RBF results become very similar to each other.

![kernel-penalty-slider](https://user-images.githubusercontent.com/46482002/223175975-7c9358c3-d306-4634-bdae-67b0d84bc260.png)

![kernel-penalty-slider-3](https://user-images.githubusercontent.com/46482002/223176010-98dd9a89-e1a4-4577-a766-0cc13c1311e1.png)

Below is a plot of the effect of penalty on number of breakpoints.
![kernel-penalty-nbps](https://user-images.githubusercontent.com/46482002/223165645-5aacc474-d4de-4bb9-9742-8e9201545d00.png)

### Bottom-up Segmentation (BUS)

PELT is a greedy algorithm that works up to the number of points whilst BUS starts with a large set of change points and deletes less significant ones. The signal is broken into chunks and merged based on how similar they are. The edges of the merged segments are where the breakpoints are.

![](https://ctruong.perso.math.cnrs.fr/ruptures-docs/build/html/_images/schema_tree.png)

Looking at the effect of penalty on the number of breakpoints and placement, it can be observed that the cost models behave quite differently. At low penalty, the breakpoints of AR achieve a similar performance to BreakpointFit with fewer breakpoints. Increasing the penalty by a small amount causes the number of AR breakpoints to drop off sharply.
The number of breakpoints for Normal appears to be less sensitive to penalty. The other models (L2, L1 and RBF) seem to drop off at a similar rate and the placement of breakpoints is pretty similar too.

![bottom-penalty-slider-2](https://user-images.githubusercontent.com/46482002/223401849-a4054eaa-e174-41ac-8faa-278c156dfd59.png)

![bottom-penalty-slider-4](https://user-images.githubusercontent.com/46482002/223401885-103b5215-89ee-4a75-a820-40c9c322b7d9.png)

If the number of breakpoints is set, we can see the general spread of each cost model. RBF, AR and Normal gets the first change at the start and the elbow at approx 40mm. L1 gets close but drifts further up the slope. Increasing the number of breakpoints to 30 helps fill in that elbow more.

![bottom-nbps-15](https://user-images.githubusercontent.com/46482002/223404805-7b2c8829-a77b-4448-8bcd-2cf06fa42541.png)

![bottom-nbps-30](https://user-images.githubusercontent.com/46482002/223404827-e013620d-aaff-4b5f-ae14-dac74dc61b4a.png)


The behaviour of the number of breakpoints with respect to penalty follows some interesting behaviour. The Normal behaviour has a very shallow slope not reaching a steady point until well outside the set range. The AR behaviour is very aggressive looking like a step function.

![bottom-penalty-nbps](https://user-images.githubusercontent.com/46482002/223193700-f1a684cc-1e27-4b0a-8ba6-254ec90a1d1c.png)

From these experiments, the cost model AR is the most attractive as it gets the locations we're intersted in with few breakpoints and minimal penalty keeping computational costs down.

### Natural Breaks

The natural breaks are the result of the Jenks-Fisher optimization algorithm which seeks to separate the data into groups so the breakpoints would be the boundary values of the groups. The fitting criteria is based on optimizing the the Goodness of the Variance.

The results were made using the ```jenks_breaks``` function rather than the class for ease of use.

![E00401009F45AF14_18080018_ST_2110_28-natural-models-nbps-13](https://user-images.githubusercontent.com/46482002/223408816-3708eeb6-a1fa-4b81-a561-850909d5baea.png)

## Comparison

![changepoint-comparison](https://user-images.githubusercontent.com/46482002/223229750-e72ab93a-3b78-4638-b88d-ef3a58ff7aa7.png)

