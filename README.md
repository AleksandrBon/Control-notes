# Note on why $u=-kx$

There is a core idea in control theory that I personally find hard to grasp from textbooks, or maybe I just overlooked it. So this note may be helpful for those who are self-studying control theory. This note presents my own self-learned understanding; it is absolutely not strict mathematics, and people who have taken or taught control classes may find it wrong or incomplete. In that case, feel free to leave corrections in the comments so that others can get it right.

Namely, if there is a dynamical system under control (I consider a linear 1D system)

$$ \dot x=f(x,u)=ax+bu, \text{  where  } a,b>0 $$

then there is really no way to obtain a “correct” control law solely from this equation. Formally and algebraically, it is possible to solve for $u$

$$ u= \frac{1}{b}​(\dot x−ax)  $$

but this is useless, because, at first, it creates a causal loop: to compute $\dot x$ we need
$u$, and to compute $u$ we need $\dot x$. And second, a deeper reason, is that a dynamical model alone does not define what “good behavior” means. It does not specify performance, handling qualities, or stability margins. These properties are target desing requirements itself.

The core idea to get out of this — and this is how it is normally done — is to postulate how we want the system to behave. That is, we postulate that the system should behave as

$$ \dot x = g(x) $$

But what is $g(x)$? In principle, one can choose any $g(x)$ that is achievable given the system dynamics and input constraints. Then, by equating the two expressions,

$$ g(x)=f(x,u),$$

we can solve for

$$ u=φ(x),$$

which no longer depends on $\dot x$ and is therefore causally correct. By applying this control law, the original system should behave like the desired, postulated one. This idea forms the basis of dynamic inversion (or feedback linearization) in control theory.

There is one particularly special choice of $g(x)$:

$$ \dot x =g(x)=−cx,c>0 $$

If we use this for our linear system, we obtain

$$ u=−\frac{a+c}{b} x=−kx, k=\frac{a+c}{b}>0 $$

One can immediately notice that this looks like the P part of a PID controller. This means that when we use a P controller, it is not just an ad-hoc or disconnected control concept. By doing so, we are postulating that we want our system to behave like a linear system with negative state (or output) feedback.

But why $\dot x =−cx$, or equivalently $u=−kx$? What makes this case so special and fundamental?

To answer this, we need to briefly dive into Lyapunov stability theory, particularly its use in constructing control laws.

When we design a controller, we want it to stabilize the system around a setpoint. In the Lyapunov framework, this requires finding a function $V(x)>0$ such that 
$\dot V(x)<0$. Lyapunov theory does not provide a recipe for finding such a function, but once one is found, stability of the dynamical system is guaranteed.

Because we are interested in controlling physical systems—and this argument was also used by Lyapunov—we can choose the system’s energy around the setpoint as a candidate Lyapunov function. This leads to a quadratic form,

$$ V(x)= \frac{1}{2} x^2 $$

Such quadratic "cost" functions are central to many control and estimation algorithms, such as LQR, Kalman filtering, and quadratic cost or value functions in reinforcement learning.

Taking the derivative gives

$$ \dot V(x) =x \dot x $$

and substituting $\dot x =−cx$ yields

$$ \dot V(x) = -c x^2 $$

which guarantees $\dot V(x) < 0$. This shows that the system $\dot x =−cx$ is stable, and that using $u = -kx$ transforms our original unstable system $\dot x =ax$ into a stable one. 

The next step for the engineer is to choose the constant $k$, which parameterizes the desired closed-loop behavior. This parameter determines how aggressively the system converges to equilibrium. In practice, its value is limited by physical constraints such as actuator dynamics and saturation.

The system $\dot x =−cx$ is not special because it models the physics of a real plant, but because it represents the simplest dynamical system whose stability can be globally certified using a quadratic Lyapunov function. By postulating this system as the desired closed-loop behavior, we explicitly define what “good behavior” means, independently of the underlying plant dynamics.

So, this was an “empirical” way to derive a control law by postulating desired dynamics. A mathematically rigorous derivation — showing why the control law should be
$u=−kx$, how to compute the optimal $k$ for a quadratic cost function, and how this relates to reinforcement learning — can be found in amazing article of Frank Lewis and co. https://share.google/BA4FcEI1qlWdVwUdj.
