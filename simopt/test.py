"""Browser smoke script for the SimOpt web UI."""

# ruff: noqa: RUF001

from importlib import import_module
from typing import Any


def run(playwright: Any) -> None:  # noqa: ANN401
    """Run recorded browser interactions against the local web UI."""
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()

    # Test 1: Check all plot functionality
    page.goto("http://localhost:5173/")
    page.get_by_role("combobox").first.select_option("ADAM")
    page.get_by_role("button", name="+ Add Solver").click()
    page.get_by_role("combobox").nth(2).select_option(
        "CNTNEWS-1 (Max Profit for Continuous Newsvendor)"
    )
    page.get_by_role("button", name="+ Add Problem").click()
    page.get_by_role("combobox").nth(2).select_option(
        "SAN-1 (Min Mean Longest Path for Stochastic Activity Network)"
    )
    page.get_by_role("button", name="+ Add Problem").click()
    page.get_by_role("combobox").nth(1).select_option("ALL")
    page.get_by_role("button", name="+ Add Plot").click()
    page.get_by_role("combobox").nth(1).select_option("MEAN")
    page.get_by_role("button", name="+ Add Plot").click()
    page.get_by_role("combobox").nth(1).select_option("QUANTILE")
    page.get_by_role("button", name="+ Add Plot").click()
    page.get_by_role("combobox").nth(1).select_option("AREA_MEAN")
    page.get_by_role("button", name="+ Add Plot").click()
    page.get_by_role("combobox").nth(1).select_option("AREA_STD_DEV")
    page.get_by_role("button", name="+ Add Plot").click()
    page.get_by_role("combobox").nth(1).select_option("SOLVE_TIME_QUANTILE")
    page.get_by_role("button", name="+ Add Plot").click()
    page.get_by_role("combobox").nth(1).select_option("SOLVE_TIME_CDF")
    page.get_by_role("button", name="+ Add Plot").click()
    page.get_by_role("combobox").nth(1).select_option("CDF_SOLVABILITY")
    page.get_by_role("button", name="+ Add Plot").click()
    page.get_by_role("combobox").nth(1).select_option("QUANTILE_SOLVABILITY")
    page.get_by_role("button", name="+ Add Plot").click()
    page.get_by_role("combobox").nth(1).select_option("AREA")
    page.get_by_role("button", name="+ Add Plot").click()
    page.get_by_role("combobox").nth(1).select_option("BOX")
    page.get_by_role("button", name="+ Add Plot").click()
    page.get_by_role("combobox").nth(1).select_option("VIOLIN")
    page.get_by_role("button", name="+ Add Plot").click()
    page.get_by_role("combobox").nth(1).select_option("TERMINAL_SCATTER")
    page.get_by_role("button", name="+ Add Plot").click()
    with page.expect_popup() as page1_info:
        page.get_by_role("button", name="Run Experiment").click()
    page1 = page1_info.value
    page1.goto("http://localhost:5173/results/20260412_182438/index.html")

    # Test 2: Edit a solver
    page.goto("http://localhost:5173/")
    page.get_by_role("combobox").first.select_option("ADAM")
    page.get_by_role("button", name="+ Add Solver").click()
    page.get_by_role("button", name="ADAM ▶ ×").click()
    page.get_by_role("button", name="Edit").click()
    page.get_by_role("textbox", name="r", exact=True).click()
    page.get_by_role("textbox", name="r", exact=True).press("ArrowRight")
    page.get_by_role("textbox", name="r", exact=True).press("ArrowRight")
    page.get_by_role("textbox", name="r", exact=True).fill("35")
    page.get_by_role("button", name="Apply Changes").click()
    page.get_by_role("button", name="ADAM ▶ ×").click()

    # Test 3: Replace a solver in edit panel with one from summary panel
    page.goto("http://localhost:5173/")
    page.get_by_role("combobox").first.select_option("ASTRODF (ASTRO-DF)")
    page.get_by_role("button", name="+ Add Solver").click()
    page.get_by_role("combobox").first.select_option("RNDSRCH (Random Search)")
    page.get_by_role("button", name="ASTRODF (ASTRO-DF) ▶ ×").click()
    page.get_by_role("button", name="Edit").click()
    page.get_by_role("button", name="Replace").click()

    # Test 4: Cancel replacement of solver in edit panel with one from summary panel
    page.goto("http://localhost:5173/")
    page.get_by_role("combobox").first.select_option("ALOE")
    page.get_by_role("button", name="+ Add Solver").click()
    page.get_by_role("combobox").first.select_option("ASTRODF (ASTRO-DF)")
    page.get_by_role("button", name="ALOE ▶ ×").click()
    page.get_by_role("button", name="Edit").click()
    page.get_by_role("button", name="Cancel").click()

    # Test 5: Partial plot (run plot for only select problems)
    page.goto("http://localhost:5173/")
    page.get_by_role("combobox").first.select_option("ADAM")
    page.get_by_role("button", name="+ Add Solver").click()
    page.get_by_role("combobox").nth(2).select_option(
        "SAN-1 (Min Mean Longest Path for Stochastic Activity Network)"
    )
    page.get_by_role("button", name="+ Add Problem").click()
    page.get_by_role("combobox").nth(2).select_option(
        "CNTNEWS-1 (Max Profit for Continuous Newsvendor)"
    )
    page.get_by_role("button", name="+ Add Problem").click()
    page.get_by_role("combobox").nth(1).select_option("MEAN")
    page.get_by_role("checkbox", name="CNTNEWS-1 (Max Profit for").check()
    page.get_by_role("button", name="+ Add Plot").click()
    with page.expect_popup() as page2_info:
        page.get_by_role("button", name="Run Experiment").click()
    _ = page2_info.value

    # Test 6: Check output log functionality
    page.goto("http://localhost:5173/")
    page.get_by_role("combobox").first.select_option("ADAM")
    page.get_by_role("button", name="+ Add Solver").click()
    page.get_by_role("combobox").nth(2).select_option(
        "SAN-1 (Min Mean Longest Path for Stochastic Activity Network)"
    )
    page.get_by_role("button", name="+ Add Problem").click()
    page.get_by_role("combobox").nth(1).select_option("MEAN")
    page.get_by_role("button", name="+ Add Plot").click()
    with page.expect_popup() as page3_info:
        page.get_by_role("button", name="Run Experiment").click()
    page3 = page3_info.value
    page3.get_by_text("Output Log ▼ Auto-scroll: ON").click()

    # Test 7: Check that experiment does not rerun when only plot is added
    page.goto("http://localhost:5173/")
    page.get_by_role("combobox").first.select_option("ADAM")
    page.get_by_role("button", name="+ Add Solver").click()
    page.get_by_role("combobox").nth(2).select_option(
        "SAN-1 (Min Mean Longest Path for Stochastic Activity Network)"
    )
    page.get_by_role("button", name="+ Add Problem").click()
    page.get_by_role("combobox").nth(1).select_option("MEAN")
    page.get_by_role("button", name="+ Add Plot").click()
    with page.expect_popup() as page5_info:
        page.get_by_role("button", name="Run Experiment").click()
    page5 = page5_info.value
    page5.goto("http://localhost:5173/results/20260412_182820/index.html")
    page.get_by_role("combobox").nth(1).select_option("BOX")
    page.get_by_role("button", name="+ Add Plot").click()
    with page.expect_popup() as page6_info:
        page.get_by_role("button", name="Run Experiment").click()
    _ = page6_info.value

    # ---------------------
    context.close()
    browser.close()


sync_api = import_module("playwright.sync_api")
with sync_api.sync_playwright() as playwright:
    run(playwright)
