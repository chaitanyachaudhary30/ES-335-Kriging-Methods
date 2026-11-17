"""Headless-browser check of the website (needs: pip install playwright, and Microsoft Edge or Chrome).

Usage
  python tests/check_site.py                  # serves docs/ locally on port 8765
  python tests/check_site.py https://...      # checks a live URL instead

Checks: no console errors, no failed requests, every data file loads, all ten sections render,
no placeholder text, every chart and map draws, and every interaction changes the page.
Screenshots go to the folder given by SHOT_DIR (default: the system temp folder).
"""
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SHOTS = Path(os.environ.get("SHOT_DIR", tempfile.gettempdir()))
PORT = 8765
problems: list[str] = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        problems.append(msg)


def main(url: str | None) -> int:
    server = None
    if url is None:
        server = subprocess.Popen([sys.executable, "-m", "http.server", str(PORT), "-d", str(ROOT / "docs")], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1.5)
        url = f"http://localhost:{PORT}/"
    try:
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(channel="msedge")
            except Exception:
                browser = p.chromium.launch(channel="chrome")
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            errors, bad = [], []
            page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on("response", lambda r: bad.append(f"{r.status} {r.url}") if r.status >= 400 else None)
            page.goto(url, wait_until="networkidle")
            page.wait_for_selector("#verdict-table tbody tr")
            page.wait_for_timeout(1500)

            # sections and placeholders
            for s in ("question", "data", "prep", "nearby", "split", "hypotheses", "methods", "results", "limits", "repro"):
                check(page.locator(f"section#{s}").count() == 1, f"section #{s} present")
            text = page.inner_text("body")
            check("Phase 2" not in text and "phase-note" not in page.content(), "no 'Phase 2' placeholder text")
            check(page.locator("td[data-verdict]").evaluate_all("els => els.every(e => e.textContent !== 'loading')"), "section 6 verdict cells filled")
            check(page.locator("#verdict-table tbody tr").count() == 7, "7 hypothesis verdicts in section 8 (H5 not shown)")
            check("lucas" not in page.content().lower(), "no mention of the unused second survey")
            check(page.locator("#res-table tbody tr").count() == 9, "results table has 9 methods")
            check(page.locator("#tune-table tbody tr").count() == 5, "tuning table has 5 rows")

            # every chart draws
            for pid in ("ph-hist", "vario-plot", "gain-plot", "dist-plot", "pred-plot", "check-plot", "skill-bar", "scatter-plot", "err-dist-plot", "density-plot", "density-lines"):
                check(page.locator(f"#{pid} .main-svg").count() > 0, f"chart #{pid} drawn")
            check(page.evaluate("state.markers.length") > 1000, "split map has its markers")
            check(page.evaluate("state.rlayer.getLayers().length") > 10, "residual map has cells")

            # every data file loads
            for f in ("site", "gemas_stats", "gemas_split_points", "gemas_predictions", "gemas_results"):
                r = page.request.get(url + f"data/{f}.json")
                check(r.ok and len(r.json()) > 0, f"data/{f}.json loads")

            # interactions
            h0 = page.inner_text("#res-headline")
            page.check("input[name=rsplit][value=random]")
            page.wait_for_timeout(300)
            check(page.inner_text("#res-headline") != h0, "results split toggle changes the headline")
            page.check("input[name=rsplit][value=spatial]")

            s0 = page.inner_text("#scatter-readout")
            page.check("input[name=ssplit][value=random]")
            page.wait_for_timeout(300)
            check(page.inner_text("#scatter-readout") != s0, "scatter split toggle changes the readout")

            n0 = page.locator("#err-dist-plot .scatterlayer .trace").count()
            page.uncheck("#err-methods input[value='Random Forest']")
            page.wait_for_timeout(300)
            check(page.locator("#err-dist-plot .scatterlayer .trace").count() == n0 - 1, "error-distance method checkbox removes a curve")
            page.select_option("#sel-emetric", "rmse")
            page.wait_for_timeout(300)
            check("RMSE" in page.inner_text("#err-dist-plot"), "error measure select works")

            m0 = page.inner_text("#res-map-readout")
            page.select_option("#sel-map-method", "Mean")
            page.wait_for_timeout(400)
            check(page.inner_text("#res-map-readout") != m0, "residual map method select updates")
            page.select_option("#sel-map-split", "random")
            page.wait_for_timeout(400)
            check(page.evaluate("state.rlayer.getLayers().length") > 10, "residual map redraws for the random split")

            d0 = page.inner_text("#density-readout")
            page.locator("#sl-frac").fill("3")
            page.wait_for_timeout(500)
            check(page.inner_text("#density-readout") != d0 and "10%" in page.inner_text("#o-frac"), "density slider updates the chart and ranking")

            g0 = page.inner_text("#gain-readout")
            page.locator("#sl-eta").fill("0.8")
            page.wait_for_timeout(300)
            check(page.inner_text("#gain-readout") != g0, "section 4 nugget slider works")
            page.check("input[name=split][value=random]")
            page.wait_for_timeout(500)
            check("Random" in page.inner_text("#split-readout"), "section 5 split toggle works")

            check(not errors, f"no console errors {errors[:3]}")
            check(not bad, f"no failed requests {bad[:3]}")

            for sec in ("methods", "results", "limits", "repro"):
                page.locator(f"section#{sec}").scroll_into_view_if_needed()
                page.wait_for_timeout(600)
            page.screenshot(path=str(SHOTS / "site_full.png"), full_page=True)
            browser.close()
    finally:
        if server:
            server.terminate()
    print("\nFAILED: " + "; ".join(problems) if problems else "\nall checks passed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
