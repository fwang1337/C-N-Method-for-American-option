# C-N Methods on American Options Pricing & IV Estimation
This project uses Crank-Nicolson Method for  pricing American options. To avoid triviality, a demonstration of pricing on put option is listed.  


-Background: Black Scholes Equation:

The Black-Scholes equation is a partial differential equation, which 
describes the price of an option over time. The key insight behind the 
equation is that one can perfectly hedge the option by 
buying and selling the underlying asses and the "bank account 
asset" (cash) in just the right way to eliminate risk.

$$\begin{equation}\frac{\partial V}{\partial t}+ \frac{1}{2}S^2\sigma^2\frac{\partial^2 V}{\partial S^2}+ (r - D)S\frac{\partial V}{\partial S} - r V = 0 \end{equation}$$

Here $V(S, T)$ is the value of the options, $S$ is the price of the 
underlying asset, $\sigma$ is the volatility of the underlying asset,
$r$ is the "risk-free" interest rate, and $D$ is the yield
(dividend paying rate) of the underlying stock.

The volatility $\sigma$ stems from an underlying assumption that 
the stock moves like a geometric Brownian motion,
$$\begin{equation}\frac{dS}{S} = \mu dt + \sigma dW. \end{equation}$$

Explicit solutions for the Black-Scholes equation,
called The Black-Scholes formulae, are known only for 
European call and put options. For other derivatives, such 
a formula doest not have to exist. However, a numerical solution is 
always possible. (Source: http://compphysics.github.io/ComputationalPhysics/doc/Projects/2020/Project5/BlackScholes/pdf/BlackScholes.tex)






*Introduction*
The market data of AAPL American put options are collected. The reason why we use put options' data for this project is that its value can have a difference in exercise value compared to its European counterparts, and in this way we can not plug analytical solutions of B-S to the answer and need to solve it numerically. We want to use finite element methods on time-dependent PDEs to transform Black-Scholes formula into its PDE form, and solve it to derive the implied volatility of each option contract on different expiration dates.

- We model stock price as a geometric Brownian Motion:
$$S_t = S_{t - \Delta t} \cdot \exp \left(\left(r - \frac{\sigma^2}{2} \right)\Delta t + \sigma \sqrt{\Delta t} W \right)$$
- Then the boundary condition for a contract with payoff max(K-S,0) at T is:
$$\begin{equation} P(0,t) = K\\ P(S,T) = max (K-S,0)  \end{equation}$$
- If we are to exercise contract at any t, the payoff is: $$\begin{equation} g(S,t) = max (K-S,0) \end{equation}$$

##{omitted paragraph for compling issue, for details please see pdf or Jupyter file}  
  
But then the issue rises: We do not know true $\sigma$ yet, so in order to infer the implied volatility  $\sigma$, we need to first have a maximum likelihood estimator $\sigma^*$ to calculate the otpion price. Then we can estimate true sigma by comparing the value of the estimated option price to the value of real price in the market. In this case, the maximum likelihood estimator is the 30-day historial volaitlity of the stock AAPL. With a simple bisection method, we can estimate $\sigma$ pretty accruately.

Algorithm: 
To find out put value at t, we implemented a CN algorithm to find the contract's value at each S, we considered extreme situations and then use a linear interpolant to calculate the contract's value at current S. Since we only know the put's terminal/boundary value, the algorithm considers the put value based on time T and then adjust to every dt. During each dt step, the CN method is performed to adjust to dS  and calculate X value. With some empirical insights from the Numeical Financial Mathematics Textbook, the k_l is set to 1.2. At each t, each S in the simulated price range, the contract X value is then determined by comparing exercising the put or hold it for another interval. After T is deducted to 0 (current time), the contract value array X with a corresponding S array for price range can be interpolated with the current stock price $S_t$. Then the put value given a historical volatility is calculated. This price is compared to the market price for determining true implied volatility for each contract with a simple bisection method with f= the algorithm with a variable  $\sigma$. 
