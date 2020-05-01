The REINFORCE algorithm is a direct differentiation of the reinforcement learning objective. What is the reinforcement learning objective, you may ask? Well, it is the following:

Eτ∼p(τ)[∑Tt=1r(st,at)]

There are a few terms to note here:

τ
is the trajectory, or the set s1,a1,s2,a2,…,st,at

when executing a policy.

st

is the current state an agent is in at timestep t.

at

is the action an agent takes at timestep t.

It makes sense that this is the reinforcement learning objective. Basically, it is the expectation over all different possible paths an agent takes of the sum of its rewards. We can directly differentiate this to get:

∇θEτ∼p(τ)[∑Tt=1r(st,at)]

=∫∇θp(τ)(∑Tt=1r(st,at))dτ

=∫p(τ)∇θlogp(τ)(∑Tt=1r(st,at))dτ

=Eτ∼p(τ)[∇θlogp(τ)∑Tt=1r(st,at)]

Now, all that’s left is to find ∇θlogp(τ)
. We know that the the only part of the trajectory parameterized by θ is the policy, or πθ(at|st)

. Thus, it is easy to work out that:

∇θlogp(τ)=∑Tt=1∇θlogπθ(at|st)

We are left with the gradient as

∇θEτ∼p(τ)[∑Tt=1r(st,at)]

=Eτ∼p(τ)[(∑Tt=1∇θlogπθ(at|st))(∑Tt=1r(st,at))]

This is the essence of the REINFORCE algorithm. By performing gradient descent on this by a Monte Carlo estimate of the expected value, we can find the optimal policy. Note: there are a couple of tricks to make policy gradient work better, such as state-dependent baselines and rewards-to-go, but all of these are variance-reduction techniques and work off of this basic algorithm.
