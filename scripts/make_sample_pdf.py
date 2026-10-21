"""Generate samples/cell_biology_notes.pdf, a small multi-page PDF used for
testing and demos. The notes are original text written for this project.

Usage (needs fpdf2 from requirements-dev.txt):
    python scripts/make_sample_pdf.py
"""

from pathlib import Path

from fpdf import FPDF

OUT = Path(__file__).resolve().parent.parent / "samples" / "cell_biology_notes.pdf"

SECTIONS = [
    (
        "1. Cell Theory",
        [
            "Cell theory is one of the foundations of biology. It has three main ideas: "
            "all living things are made of one or more cells, the cell is the basic unit "
            "of structure and function in living things, and all cells come from "
            "pre-existing cells.",
            "The first two ideas came from the work of Matthias Schleiden, who studied "
            "plants, and Theodor Schwann, who studied animals, in the late 1830s. Rudolf "
            "Virchow added the third idea in 1855 with the phrase 'omnis cellula e cellula'.",
            "Cells are usually microscopic. A typical animal cell is between 10 and 30 "
            "micrometres across, while most bacteria are only 1 to 5 micrometres long. "
            "Their small size keeps the surface area to volume ratio high, which lets "
            "nutrients and waste move in and out quickly enough.",
        ],
    ),
    (
        "2. Prokaryotic and Eukaryotic Cells",
        [
            "Prokaryotic cells, such as bacteria and archaea, do not have a nucleus. Their "
            "DNA is a single circular chromosome found in a region called the nucleoid. "
            "Many prokaryotes also carry small rings of DNA called plasmids, which often "
            "contain genes for antibiotic resistance.",
            "Eukaryotic cells, found in animals, plants, fungi and protists, keep their DNA "
            "inside a membrane-bound nucleus. They also contain other membrane-bound "
            "organelles that divide the cell into compartments with different jobs.",
            "Both cell types have a cell membrane, cytoplasm and ribosomes. Prokaryotic "
            "ribosomes are smaller (70S) than the ribosomes in the eukaryotic cytoplasm "
            "(80S), a difference that some antibiotics use to target bacteria.",
        ],
    ),
    (
        "3. Organelles and Their Functions",
        [
            "The nucleus stores the cell's DNA and controls gene expression. It is "
            "surrounded by a double membrane called the nuclear envelope, which has pores "
            "that let RNA and proteins pass through. The nucleolus inside the nucleus "
            "makes ribosomal RNA.",
            "Mitochondria are often called the powerhouse of the cell because they produce "
            "most of its ATP through aerobic respiration. They have their own circular DNA "
            "and a folded inner membrane; the folds are called cristae and increase the "
            "surface area for respiration.",
            "Ribosomes build proteins by reading messenger RNA. Free ribosomes make "
            "proteins used inside the cell, while ribosomes attached to the rough "
            "endoplasmic reticulum make proteins that will be exported or placed in "
            "membranes.",
            "The smooth endoplasmic reticulum makes lipids and helps detoxify drugs. The "
            "Golgi apparatus modifies, sorts and packages proteins into vesicles. "
            "Lysosomes contain digestive enzymes that break down worn-out organelles and "
            "material taken in by the cell.",
            "Plant cells also have chloroplasts, where photosynthesis happens, a large "
            "central vacuole that stores water and keeps the cell firm, and a cell wall "
            "made of cellulose.",
        ],
    ),
    (
        "4. The Cell Membrane and Transport",
        [
            "The cell membrane is described by the fluid mosaic model: a phospholipid "
            "bilayer with proteins floating in it. Phospholipids have hydrophilic heads "
            "that face the water on both sides and hydrophobic tails that face inward. "
            "Cholesterol sits between the phospholipids and helps control fluidity.",
            "Diffusion is the net movement of particles from an area of higher "
            "concentration to an area of lower concentration. It is passive, meaning it "
            "does not need energy from the cell. Small non-polar molecules such as oxygen "
            "and carbon dioxide diffuse straight through the bilayer.",
            "Facilitated diffusion uses channel or carrier proteins to help larger or "
            "charged particles cross the membrane, still moving down their concentration "
            "gradient. Osmosis is the diffusion of water across a partially permeable "
            "membrane from a dilute solution to a more concentrated one.",
            "Active transport moves substances against their concentration gradient and "
            "needs energy from ATP. The sodium-potassium pump is a classic example: each "
            "cycle pumps three sodium ions out of the cell and two potassium ions in.",
        ],
    ),
    (
        "5. Cell Division",
        [
            "Mitosis produces two genetically identical daughter cells and is used for "
            "growth and repair. Its stages are prophase, metaphase, anaphase and "
            "telophase, usually followed by cytokinesis, which splits the cytoplasm.",
            "During prophase the chromosomes condense and the spindle starts to form. In "
            "metaphase the chromosomes line up along the middle of the cell. In anaphase "
            "the sister chromatids are pulled to opposite poles, and in telophase new "
            "nuclear envelopes form around each set of chromosomes.",
            "Meiosis makes gametes. It involves two divisions and produces four cells, "
            "each with half the original number of chromosomes. Crossing over and "
            "independent assortment during meiosis create genetic variation.",
        ],
    ),
    (
        "6. Cellular Respiration",
        [
            "Aerobic respiration releases energy from glucose using oxygen. The overall "
            "equation is glucose + oxygen -> carbon dioxide + water, with the energy "
            "stored as ATP.",
            "Glycolysis happens in the cytoplasm and splits one glucose molecule into two "
            "pyruvate molecules, giving a small net gain of 2 ATP. The link reaction and "
            "the Krebs cycle take place in the mitochondrial matrix and release carbon "
            "dioxide.",
            "Most ATP is made by oxidative phosphorylation on the inner mitochondrial "
            "membrane, where the electron transport chain uses oxygen as the final "
            "electron acceptor. In total, one glucose molecule yields roughly 30 to 32 "
            "ATP.",
            "Without oxygen, cells rely on anaerobic respiration. Human muscle cells "
            "produce lactate, while yeast produces ethanol and carbon dioxide. Anaerobic "
            "respiration releases far less ATP than aerobic respiration.",
        ],
    ),
]


class NotesPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "I", 9)
        self.cell(0, 8, "EduPulse sample notes - Cell Biology", align="R")
        self.ln(12)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 9)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def build() -> Path:
    pdf = NotesPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_title("Cell Biology - Study Notes")
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "Cell Biology: Study Notes", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    for heading, paragraphs in SECTIONS:
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 9, heading, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 12)
        for paragraph in paragraphs:
            pdf.multi_cell(0, 7, paragraph, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
        pdf.ln(3)

    OUT.parent.mkdir(exist_ok=True)
    pdf.output(str(OUT))
    return OUT


if __name__ == "__main__":
    path = build()
    print(f"Wrote {path}")
