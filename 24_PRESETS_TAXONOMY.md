# Envisage: 24 Surgical Presets Taxonomy

This document lists the 24 empirical presets (8 rhinoplasty, 8 blepharoplasty, 8 rhytidectomy) used in the Envisage pipeline to guide the AI's surgical edits. Each preset is visually distinct, landmark-detectable, and acts as a localized constraint on the diffusion model.

## Rhinoplasty (8 Presets)
*Grounded in Rollin K. Daniel, "Mastering Rhinoplasty" (2nd Ed).*
1. **Dorsal Hump Removal** (`dorsal_hump_reduction`): #1 most common request (~70% of patients). Bone rasp plus cartilage scissors.
2. **Tip Definition** (`tip_definition`): 'As the tip goes so goes the result.' Domal suture plus tip refinement grafts.
3. **Tip Narrowing** (`tip_narrowing`): Reduce dome width. L1 interdomal suture, L2 domal creation, L3 dome excision.
4. **Bridge Narrowing** (`dorsal_narrowing`): Lateral osteotomies narrow the bony vault. Indicated when alar-width to intercanthal-distance ratio exceeds 1.05.
5. **Tip Rotation (Upward)** (`tip_rotation_up`): Correct drooping or ptotic tip. Tip-position suture plus caudal septal resection.
6. **Alar Base Narrowing** (`alar_base_narrowing`): Reduce wide nostril base. Alar width should approximate intercanthal distance.
7. **Dorsal Straightening** (`dorsal_straightening`): Asymmetric osteotomies plus septal relocation correct a crooked or deviated dorsum.
8. **Nose Shortening** (`nose_shortening`): Caudal septal resection plus tip rotation for an over-long nose.

## Blepharoplasty (8 Presets)
1. **Upper Lid Skin Excision** (`upper_skin_excision`): #1 most common. Remove redundant upper-lid skin (dermatochalasis).
2. **Supratarsal Crease Restoration** (`crease_restoration`): Restore or create defined upper-lid crease. Canonical: 8-10mm female, 6-8mm male.
3. **Upper Lid De-hooding** (`upper_dehooding`): Remove excess skin covering the lid fold. Most visible improvement.
4. **Eyelid Symmetry Correction** (`lid_symmetry`): Correct asymmetric lid-crease height or hooding amount between eyes.
5. **Upper Medial Fat Pad Reduction** (`fat_pad_reduction`): Remove herniated medial fat pad causing upper-lid fullness.
6. **Lower Lid Bag Reduction** (`lower_bag_reduction`): Remove or reposition herniated lower-lid fat pads.
7. **Tear Trough Smoothing** (`tear_trough_smoothing`): Fill or smooth nasojugal groove via fat transposition or filler.
8. **Periorbital Fine-line Softening** (`crow_feet_softening`): Mild smoothing of lateral periorbital lines. Not full removal.

## Rhytidectomy (Facelift) (8 Presets)
1. **Jawline Straightening** (`jawline_straightening`): #1 priority. SMAS lift creates a clean jawline with straight-edge contour.
2. **Jowl Elimination** (`jowl_elimination`): #2 priority. Eliminate tissue ptosis below the jawline.
3. **Neck Skin Smoothing** (`neck_smoothing`): Smooth anterior neck without changing size or proportions. Texture-only edit.
4. **Marionette Line Softening** (`marionette_softening`): SMAS repositioning reduces marionette creases from mouth to jaw.
5. **Platysmal Band Removal** (`platysmal_band_removal`): Remove visible vertical neck bands. Smooth texture only.
6. **Pre-jowl Sulcus Correction** (`prejowl_correction`): Fill the depression anterior to the jowl for continuous contour.
7. **Submental Definition** (`submental_definition`): Define cervicomental angle while keeping neck size unchanged.
8. **Lower Nasolabial Softening** (`nasolabial_softening`): Only the LOWER portion of the nasolabial fold. Upper requires filler.
