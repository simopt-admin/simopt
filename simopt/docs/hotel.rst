
Hotel Revenue Management
========================

Most of the revenue for a hotel comes from guests staying in its rooms; since this
resource is constrained by the capacity in each hotel, let it be :math:`C`, one goal
may be to maximize expected revenue given the distribution of demands.

Assume a given hotel has only two rates: rack rate and discount rate, which pay :math:`p_f`
and :math:`p_d` per night, respectively. Furthermore, let each different combination
of length of stay, arrival date and rate paid be a "product" so that the following
56 products are available to satisfy one week's worth of capacity (14 arriving Monday,
12 arriving Tuesday, ..., 2 Arriving Sunday):

1. One night stay, rack rate arriving Monday
2. One night stay, discount rate arriving Monday
3. Two night stay, rack rate arriving Monday
4. Two night stay, discount rate arriving Monday
5. ...
55. One night stay, rack rate arriving Sunday
56. One night stay, discount rate arriving Sunday

For a given stay, the hotel collects revenue equal to the (rate paid) x (length of stay).
Lastly, let the arrival processes for each product be a stationary Poisson process with
rate :math:`\lambda_i`, noting that orders for a Monday night stay stop arriving at
3 AM Tuesday night, for a Tuesday night stay at 3 AM Wednesday and so on.

With this in mind, our goal is to find a set of booking limits (:math:`b_1, ..., b_{56}`)
that maximizes expected revenue. Under this model, the booking limits are controls
that limit the amount of capacity that can be sold to any particular product; i.e.,
they represent the maximum number of requests of product :math:`i` we are willing to
accept. We must note that the booking limits do not represent the number of rooms
reserved for each product, rather, they represent the number of rooms available to
this product and all products that use the same resources and have a higher booking limit.
For example, if we have five products and all of them require the same resource (say
:math:`C = 10`) and their corresponding booking limits are :math:`b_1 = 10, b_2 = 8,
b_3 = 4, b_4 = 2, b_5 = 1`, we know we can only take 1 request for product 5, 2 requests
for product 4 and so on. However, this **does not mean that 2 rooms will be saved**
until 2 requests for product 4 arrive; **rather**, it means that, **out of all requests
accepted, at most 2 can be of product 4**. Note also that the maximum number of requests
accepted in this case would be 10, as they all use the same resource, which has :math:`C = 10`.
Doing this ensures that those products with higher booking limits are always accepted
if capacity is available while also accounting for the interconnectedness of the system.

Now, once the booking limits are set, a request for product :math:`i` is accepted if
and only if :math:`b_i > 0` and rejected otherwise. When a request for product :math:`i` is
accepted, all of the booking limits that require the resources used by product :math:`i`
must be updated to account for the decrease in available resources. For example,
if a request for a 3-night stay arriving Monday is accepted, all products using a night
on either Monday, Tuesday, or Wednesday must have their booking limits decreased by one.

With a little thought, we may see that :math:`b_i \leq C` (to avoid overbooking) and
that the highest booking limit must equal capacity (we want to rent as many rooms as
possible without going over capacity, thus, all rooms must be available to at least one
product). Furthermore, since requests are only accepted when rooms are available
(:math:`b_i > 0`), we are guaranteed to never go over capacity.

In summary, a booking limit represents the maximum number of requests of product :math:`i`
that we are willing to accept given that we start with full availability. As soon as
a request is accepted, available capacity changes and booking limits must be updated
to account for this change. Although our interest is in modeling the full 56 products
to find the optimal set of booking limits, to illustrate how booking limits are updated,
one may look at the following, small-scale example:

Assume a hotel offers only the following 5 products:
1. Two night stay arriving Monday.
2. Two night stay arriving Tuesday.
3. Two night stay arriving Wednesday.
4. Three night stay arriving Wednesday.
5. Two night stay arriving Thursday.

If the booking limits for each product are :math:`b_1 = 10, b_2 = 8, b_3 = 4, b_4 = 7, 
b_5 = 1` and the following requests are received, the booking limits would be updated
in the following way as decisions to accept or reject a given order are made:

.. image:: hotel.PNG
  :alt: The HOTEL table has failed to display
  :width: 800

Note that, in this case, product 1 has a final booking limit of 9 as only one room
has been sold on either Monday or Tuesday, which means that 9 rooms are still available
on Monday night.

To simplify this, one may create a binary matrix A showing which products use which
resources. Thus, we will let each row be a resource available and each column a product,
having a 1 in entry :math:`(i,j)` if product :math:`j` uses resource :math:`i`, and 0 
otherwise. Then, if we accept a request for product :math:`i`, we must update the booking
limits of all products :math:`j` such that :math:`A_j^T \cdot A_i \geq 1` (they share
at least one of the resources). For this small example, we have:

.. image:: hotel2.PNG
  :alt: The HOTEL matrix has failed to display
  :width: 300

*Recommended Parameter Settings:* :math:`C = 100, p_d = 100, p_f = 200`. Take :math:`\lambda_i =
\frac{1}{168}, \frac{2}{168}, \frac{3}{168}, \frac{2}{168}, \frac{1}{168}, \frac{0.5}{168},
\frac{0.25}{168}` for 1-night, 2-night, ..., 7-night stay respectively.

*Starting Solution(s):* :math:`b_i = C`. If multiple solutions are needed, let each
:math:`b_i` be distributed Uniformly :math:`(0,C)`.

*Measurement of Time:* Start taking orders one week in advance of the start of simulation
(:math:`t = -168` hours) up until Sunday night.

*Recommended Budgets:* None.

*Optimal Solution:* Unknown.