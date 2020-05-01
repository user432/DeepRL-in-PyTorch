The REINFORCE algorithm is a direct differentiation of the reinforcement
learning objective. What is the reinforcement learning objective, you
may ask? Well, it is the following:

\$ E\_{\tau \sim p(\tau)} [\sum*{t = 1}\^{T} r(s*t, a\_t)] \$

There are a few terms to note here:

$\tau $ is the trajectory, or the set \$ {s\_1, a\_1, s\_2, a\_2, …,
s\_t, a\_t} \$ when executing a policy. $s_t$ is the current state an
agent is in at timestep t. $a_t $ is the action an agent takes at
timestep t.

It makes sense that this is the reinforcement learning objective.
Basically, it is the expectation over all different possible paths an
agent takes of the sum of its rewards. We can directly differentiate
this to get:

$\nabla_{\theta} E_{\tau \sim p(\tau)} [ \sum_{t = 1}^{T} r(s_t, a_t) ] $
$= \int \nabla_{\theta} p(\tau) (\sum_{t = 1}^{T} r(s_t, a_t) ) d\tau$
\$ = \int p(\tau) \nabla*{\theta} \log p(\tau)(\sum*{t = 1}\^{T} r(s\_t,
a\_t)) d\tau\$ \$ = E\_{\tau \sim p(\tau)} [\nabla*{\theta} \log p(\tau)
\sum*{t = 1}\^{T} r(s\_t, a\_t)] \$

Now, all that’s left is to find $\nabla_{\theta} \log p(\tau) $. We know
that the the only part of the trajectory parameterized by $\theta $ is
the policy, or \$ \pi*{\theta}(a*t | s\_t) \$. Thus, it is easy to work
out that:

$\nabla_{\theta} \log p(\tau) = \sum_{t = 1}^{T} \nabla_{\theta}\log \pi_{\theta}(a_t | s_t) $

We are left with the gradient as
$\nabla_{\theta} E_{\tau \sim p(\tau)} [ \sum_{t = 1}^{T} r(s_t, a_t) ] $
$= E_{\tau \sim p(\tau)} [(\sum_{t = 1}^{T} \nabla_{\theta}\log \pi_{\theta}(a_t | s_t)) (\sum_{t = 1}^{T} r(s_t, a_t) )] $

This is the essence of the REINFORCE algorithm. By performing gradient
descent on this by a Monte Carlo estimate of the expected value, we can
find the optimal policy. Note: there are a couple of tricks to make
policy gradient work better, such as state-dependent baselines and
rewards-to-go, but all of these are variance-reduction techniques and
work off of this basic algorithm.

So, the pseudo code goes as:
