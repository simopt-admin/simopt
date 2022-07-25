directory module
================
The `directory.py <directory.html>`_ module contains a current listing of models, problems, and solvers in the library. This list is continuously updated with the new implementations and additions.

Each problem has an abbreviation indicating which types of solver is compatible to solve it. The letters in the abbreviation stand for:
    <table>
        <tr>
          <th> Objective </th>
          <th> Constraint </th>
          <th> Variable </th>
          <th> Gradient </th>
        </tr>
        <tr>
          <td> Single (S) </td>
          <td> Unconstrained (U) </td>
          <td> Discrete (D) </td>
          <td> Gradient Observations Available (G) </td>
        </tr>
      <tr>
          <td> Multiple (M) </td>
          <td> Box (B) </td>
          <td> Continuous (C) </td>
          <td> Gradient Observations Not Available (N) </td>
        </tr>
      <tr>
          <td>  </td>
          <td> Deterministic (D) </td>
          <td> Mixed (M)  </td>
          <td>  </td>
        </tr>
      <tr>
          <td>  </td>
          <td> Stochastic (S) </td>
          <td> </td>
          <td>  </td>
        </tr>
    </table>

.. automodule:: directory
   :members:
   :undoc-members:
   :show-inheritance:
