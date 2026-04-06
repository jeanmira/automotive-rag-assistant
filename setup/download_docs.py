"""
Download public automotive PDFs for the RAG pipeline.
Fetches documents from AUTOSAR, ISO 26262, cybersecurity, and OBD-II sources
into data/ organized by category. Skips files already downloaded.
"""

import os
import sys
import time
import requests
from pathlib import Path

DATA_DIR = Path(os.path.join(os.path.dirname(__file__), "..", "data"))

HEADERS = {"User-Agent": "Mozilla/5.0 (automotive-rag-assistant research tool)"}

# public documents -- no auth required
DOCUMENT_CATALOG = {
    "autosar": [
        {
            "name": "AUTOSAR_CP_LayeredSoftwareArchitecture",
            "url": "https://www.autosar.org/fileadmin/standards/R25-11/CP/AUTOSAR_CP_EXP_LayeredSoftwareArchitecture.pdf",
            "description": "AUTOSAR Classic Platform layered architecture overview",
        },
        {
            "name": "AUTOSAR_TR_Methodology",
            "url": "https://www.autosar.org/fileadmin/standards/R22-11/CP/AUTOSAR_TR_Methodology.pdf",
            "description": "AUTOSAR development methodology and workflow",
        },
        {
            "name": "AUTOSAR_Introduction_Part1",
            "url": "https://www.autosar.org/fileadmin/user_upload/AUTOSAR_Introduction_PDF/AUTOSAR_EXP_Introduction_Part1.pdf",
            "description": "AUTOSAR partnership overview and standardization goals",
        },
    ],
    "functional_safety": [
        {
            "name": "NXP_ISO26262_IEC61508_Overview",
            "url": "https://community.nxp.com/pwmxy87654/attachments/pwmxy87654/tech-days/160/1/AMF-AUT-T2713.pdf",
            "description": "NXP overview of ISO 26262 and IEC 61508 functional safety standards",
        },
        {
            "name": "Exida_Automotive_ISO26262",
            "url": "https://www.exida.com/marketing/automotive-iso26262.pdf",
            "description": "exida introduction to automotive functional safety and ISO 26262",
        },
        {
            "name": "ROHM_ISO26262_WhitePaper",
            "url": "https://fscdn.rohm.com/en/products/databook/white_paper/iso26262_wp-e.pdf",
            "description": "ROHM white paper on ISO 26262 for modern road vehicles",
        },
    ],
    "cybersecurity": [
        {
            "name": "NREL_Vehicle_Cybersecurity_Threats",
            "url": "https://docs.nrel.gov/docs/fy19osti/74247.pdf",
            "description": "NREL report on vehicle cybersecurity threats and CAN bus vulnerabilities",
        },
        {
            "name": "UNECE_ENISA_CyberSecurity_SmartCars",
            "url": "https://wiki.unece.org/download/attachments/42041673/TFCS-03-09e%20ENISA%20Cyber%20Security%20and%20Resilience%20of%20smart%20cars.pdf?api=v2",
            "description": "ENISA good practices for securing smart cars against cyber threats",
        },
        {
            "name": "UNECE_OICA_Cybersecurity_Regulations",
            "url": "https://wiki.unece.org/download/attachments/154665132/W2P1%20OICA.pdf?version=1&modificationDate=1643908534916&api=v2",
            "description": "OICA presentation on UN automotive cybersecurity regulations",
        },
    ],
    "diagnostics": [
        {
            "name": "CARB_OBD2_Regulation",
            "url": "https://ww2.arb.ca.gov/sites/default/files/barcu/regact/2021/obd2021/isor.pdf",
            "description": "California ARB OBD-II regulation with UDS protocol requirements",
        },
        {
            "name": "DigiKey_Getting_Started_OBD-II",
            "url": "https://mm.digikey.com/Volume0/opasdata/d220001/medias/docus/1268/Getting_Started_with_OBD-II.pdf",
            "description": "Getting started with OBD-II on-board diagnostics overview",
        },
    ],
}


def download_document(name, url, category):
    """Download a single PDF. Skips if already on disk."""
    dest = DATA_DIR / category / f"{name}.pdf"
    dest.parent.mkdir(parents=True, exist_ok=True)

    # already have it -- skip
    if dest.exists():
        size_kb = dest.stat().st_size // 1024
        print(f"  skip  {name} ({size_kb}kb)")
        return True

    print(f"  fetch {name}...", end=" ", flush=True)

    try:
        # stream to avoid loading large PDFs into memory at once
        response = requests.get(url, headers=HEADERS, timeout=60, stream=True)
        response.raise_for_status()

        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        size_kb = dest.stat().st_size // 1024
        print(f"ok ({size_kb}kb)")

        # small delay to be respectful to public servers
        time.sleep(1.5)
        return True

    except Exception as e:
        print(f"failed: {e}")
        # clean up partial downloads
        dest.unlink(missing_ok=True)
        return False


def main():
    """Download all documents from the catalog, grouped by category."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"downloading documents to {DATA_DIR.resolve()}\n")

    results = {"ok": [], "failed": []}

    for category, docs in DOCUMENT_CATALOG.items():
        print(f"[{category}]")
        for doc in docs:
            ok = download_document(doc["name"], doc["url"], category)
            target = results["ok"] if ok else results["failed"]
            target.append(doc["name"])
        print()

    # summary
    print(f"downloaded: {len(results['ok'])} documents")
    if results["failed"]:
        print(f"failed: {', '.join(results['failed'])}")
        print("check network or try running again -- some servers rate limit")
        sys.exit(1)

    print("\nnext step: run the app and click 'index documents' to build the vector store")


if __name__ == "__main__":
    main()
