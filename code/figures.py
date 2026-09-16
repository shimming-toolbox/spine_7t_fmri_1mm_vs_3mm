import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from nibabel.processing import resample_from_to
import seaborn as sns


#####################################################
class Figures_main:
    """
    The Postprocess_main class is used to setup the Post-processing path and execute the Post-processing steps.

    Attributes
    ----------
    config : dict
        Defining all the parameters of the analysis including the path to the raw data, the participants to analyze, the design of the experiment, and the preprocessing parameters
    IDs : list
        List of participant IDs to process (e.g., ['A001', 'A002'])
    verbose : bool
        Whether to print information during the each step (default: True)
    """

    def __init__(self, config, IDs=None,verbose=True):
        if IDs==None:
            raise ValueError("Please provide the participant ID (e.g., _.stc(ID='A001')).")
        
        # Class attributes -------------------------------------------------------------------------------------
        self.config = config # load config info
        self.participant_IDs= IDs # list of the participants to analyze
        self.raw_dir = self.config["raw_dir"]  # directory of the raw data
        self.derivatives_dir = os.path.join(self.config["raw_dir"], self.config["derivatives_dir"])  # directory of the derivatives data
        self.first_level_dir = os.path.join(self.config["raw_dir"], self.config["first_level"]["dir"])  # directory of the derivatives data
        self.second_level_dir = os.path.join(self.config["raw_dir"], self.config["second_level"]["dir"])  # directory of the second-level analysis data
        self.manual_dir = os.path.join(self.config["raw_dir"], self.config["manual_dir"])  # directory of the manual corrections
        self.first_level_fig=os.path.join(self.config["raw_dir"], self.config["figures_dir"]["main_dir"],"first_level") 
        self.second_level_fig=os.path.join(self.config["raw_dir"], self.config["figures_dir"]["main_dir"],"second_level")

        os.makedirs(self.first_level_fig,exist_ok=True)
        os.makedirs(self.second_level_fig,exist_ok=True)
     
    def plot_first_level_maps(self, i_fnames=None, output_fname=None,titles=["3mm","1mm smoothed"],cmap="autumn",stat_min=1.6, stat_max=4,background_fname=None,mask_fname=None, underlay_fname=None,task_name=None,plot_mip=True, participant_ids=None, verbose=True, redo=False,n_cols=5):
        """
        Plot first-level statistical maps for multiple participants and contrasts in a grid layout.

        To do: add spinal levels in the coronal view 
        """
        if output_fname is None:
            output_fname = os.path.join(self.first_level_dir.split("sub-")[0], f"first_level_maps_n{len(i_fnames)}_all.png")
        if i_fnames is None or len(i_fnames) == 0:
            raise ValueError("i_fnames_pair is empty")

        if not os.path.exists(output_fname) or redo:
            n_subjects = len(i_fnames)
            n_participant_rows = (n_subjects + n_cols - 1) // n_cols  # number of participant rows
            n_rows = n_participant_rows * 3  # coronal, axial, gap
            n_actual_cols = min(n_subjects, n_cols)
            total_cols = (n_cols * 3) - 1  # 2 maps + 1 spacer per participant except for the last one

            # --- Load template, mask, and underlay ---
            template_img = nib.as_closest_canonical(nib.load(background_fname))
            template_data = template_img.get_fdata()
            mask_data = None
            if mask_fname is not None:
                mask_img = nib.load(mask_fname)
                mask_data = nib.as_closest_canonical(mask_img).get_fdata()

            underlay_data = None
            if underlay_fname is not None:
                underlay_data = nib.as_closest_canonical(nib.load(underlay_fname)).get_fdata()

            # --- Figure and gridspec ---
            fig_width = 7 #max paper width is 7 inches
            left, right, top, bottom = 0.01, 0.90, 0.90, 0.01

            # Coronal crop is (x_max-x_min) x (z_max-z_min) = 70 x 220; axial crop is
            # 2*crop_x x 2*crop_y = 60 x 60. Both are shown with aspect='equal' at the same
            # column width, so (a) their height_ratios must be in the same ratio as their
            # own height/width, or one gets letterboxed and the two rows' left/right edges
            # won't line up, and (b) fig_height must be picked so a height-ratio-unit's
            # absolute size actually matches a width-ratio-unit's -- otherwise aspect='equal'
            # still letterboxes (just equally on both rows), leaving unwanted blank margin
            # on either side of every panel despite a small wspace/spacer.
            coronal_aspect = 220 / 70
            axial_aspect = 60 / 60
            gap_ratio = 0.6
            spacer_ratio = 0.1

            height_ratios = []
            for _ in range(n_participant_rows):
                height_ratios += [coronal_aspect, axial_aspect, gap_ratio]

            width_ratios = []
            for i in range(n_cols):
                width_ratios += [1, 1]  # two map columns
                if i != n_cols - 1:     # add spacer except after last participant
                    width_ratios += [spacer_ratio]

            col_width_units = n_cols * 2 + (n_cols - 1) * spacer_ratio
            row_height_units = n_participant_rows * (coronal_aspect + axial_aspect + gap_ratio)
            # Require (height of 1 row-unit) == (width of 1 column-unit), in inches.
            fig_height = fig_width * (right - left) / col_width_units * row_height_units / (top - bottom)

            fig = plt.figure(figsize=(fig_width, fig_height))
            fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)

            gs = fig.add_gridspec(nrows=len(height_ratios), ncols=total_cols,
                            height_ratios=height_ratios,
                            width_ratios=width_ratios,
                            hspace=0.01,wspace=0.002)

            for subj_idx, maps in enumerate(i_fnames):
                col_idx = subj_idx % n_cols
                row_participant = subj_idx // n_cols
                row_start = (subj_idx // n_cols) * 3
                col_start = (subj_idx % n_cols) * 3   # 2 for maps, 1 for spacer

                for map_idx, i_fname in enumerate(maps):
                    if i_fname is None:
                        ax = fig.add_subplot(gs[row_start, col_start + map_idx])
                        ax.axis("off")   # empty panel
                        continue

                    x_min, x_max = 35, 105
                    # z indices into template/PAM50_t2.nii.gz as bundled in this repo, which
                    # is NOT the full official PAM50 template (982 z-slices) -- it's already
                    # cropped down to 441 z-slices (0.5mm), spanning PAM50 physical z = -286.8mm
                    # (index 0) to -66.8mm (index 440). z=130-350 here is therefore PAM50
                    # z=-221.8mm to -111.8mm. (Vertebral/spinal level at these coordinates is
                    # not annotated here -- a prior hardcoded level table was removed as
                    # unverified, see d134a3f.)
                    z_min, z_max = 130, 350
                    statmap_img = nib.as_closest_canonical(nib.load(i_fname))
                    statmap_data = statmap_img.get_fdata()
                    if mask_data is not None:
                        mask_resampled = resample_from_to(mask_img, statmap_img, order=0)  # nearest-neighbor for mask
                        mask_data = mask_resampled.get_fdata() > 0  # boolean
                        statmap_data = np.where(mask_data, statmap_data, 0)
            
                    stat_thresh = np.where(statmap_data > stat_min, statmap_data, 0)

                    # Axial slice below: its anatomical background is the geometric mid-point
                    # of the (PAM50) Z axis -- not an activation peak or anatomical landmark,
                    # just a fixed, consistent level across subjects/conditions. But (when
                    # plot_mip=True, the default) the BOLD overlay itself is a max-intensity
                    # projection across the *entire* Z axis (see mip_axi below), same as the
                    # coronal view -- so it is not specific to this single slice, and no
                    # z-position indicator is drawn on the coronal view for it.
                    z_slice = statmap_data.shape[2] // 2 if plot_mip else 260

                    # --- Coronal (top row) ---
                    if plot_mip:
                        y_slice = statmap_data.shape[1] // 2
                        mip_cor = np.max(stat_thresh, axis=1)
                        mip_cor = mip_cor[x_min:x_max,z_min:z_max]
                    else:
                        y_slice = 69
                        mip_cor = stat_thresh
                        mip_cor = mip_cor[x_min:x_max,y_slice, z_min:z_max]
                    mip_cor = np.where(mip_cor > stat_min, mip_cor, np.nan)
                    mip_cor=mip_cor.T
                    template_cor = template_data[x_min:x_max, y_slice, z_min:z_max].T

                    ax_cor = fig.add_subplot(gs[row_start, col_start + map_idx])
                    ax_cor.imshow(template_cor, cmap="gray", origin="lower",aspect='equal')
                    if underlay_data is not None:
                        ax_cor.imshow(underlay_data[x_min:x_max, y_slice, z_min:z_max].T, cmap="gray", origin="lower",aspect='equal')

                    ax_cor.imshow(mip_cor, cmap=cmap, origin="lower", vmin=stat_min, vmax=stat_max,aspect='equal')
                    ax_cor.axis("off")

                    if map_idx == 0:
                        # Center of the 2-map block, computed from the actual gridspec
                        # geometry (not a visual guess): midpoint between this axis's own
                        # left edge and the second map's right edge, expressed as a
                        # fraction of this axis's own width (transAxes).
                        x_center = 1.001
                        xmax_line = 2.001
                        y_top = 1.1
                        if participant_ids is not None:
                            subj_label = participant_ids[subj_idx]
                        else:
                            subj_label = subj_idx + 1
                        ax_cor.text(x_center, y_top, f"sub-{subj_label}", ha='center', va='bottom', fontsize=8, fontweight='black', transform=ax_cor.transAxes, fontname="Arial")
                        line_y = 1.08
                        ax_cor.hlines(y=line_y, xmin=0, xmax=xmax_line, colors='black', linewidth=0.8, transform=ax_cor.transAxes, clip_on=False)

                        ax_cor.set_title(titles[0], color="black",  fontsize=6, fontname="Arial", y=0.99)
                    if map_idx == 1:
                        ax_cor.set_title(titles[1], color="black",  fontsize=6, fontname="Arial", y=0.99)

                    # Orientation labels only for first participant
                    if subj_idx == 0 and map_idx == 0:
                        ax_cor.text(0.05, 0.05, "L", transform=ax_cor.transAxes, color="white", fontsize=3.5, ha="left", va="bottom")
                        ax_cor.text(0.95, 0.05, "R", transform=ax_cor.transAxes, color="white", fontsize=3.5, ha="right", va="bottom")

                    # --- Axial (bottom row) ---
                    row_axi = row_start + 1

                    # Crop for smaller axial view
                    crop_x = 30
                    crop_y = 30
                    x0 = statmap_data.shape[0] // 2
                    y0 = statmap_data.shape[1] // 2
                    x_min, x_max = x0 - crop_x, x0 + crop_x
                    y_min, y_max = y0 - crop_y, y0 + crop_y
                    template_axi = template_data[x_min:x_max, y_min:y_max, z_slice].T
                    
                    if plot_mip:
                        stat_crop = stat_thresh[x_min:x_max, y_min:y_max, :]
                        mip_axi = np.max(stat_crop, axis=2).T
                    else:
                        stat_crop = stat_thresh[x_min:x_max, y_min:y_max, z_slice]
                        mip_axi=stat_crop.T
                    mip_axi = np.where(mip_axi > stat_min, mip_axi, np.nan)

                    ax_axi = fig.add_subplot(gs[row_start + 1, col_start + map_idx])
                    ax_axi.imshow(template_axi, cmap="gray", origin="lower",aspect='equal')
                    if underlay_data is not None:
                        ax_axi.imshow(underlay_data[x_min:x_max, y_min:y_max, z_slice].T,
                                    cmap="gray", alpha=0.3, origin="lower")
                    ax_axi.imshow(mip_axi, cmap=cmap, origin="lower", vmin=stat_min, vmax=stat_max,aspect='equal')
                    ax_axi.axis("off")

                    if subj_idx == 0 and map_idx == 0:
                        ax_axi.text(0.02, 0.5, "L", transform=ax_axi.transAxes, color="white", fontsize=3.5, ha="left", va="center")
                        ax_axi.text(0.98, 0.5, "R", transform=ax_axi.transAxes, color="white", fontsize=3.5, ha="right", va="center")
                        ax_axi.text(0.5, 0.98, "A", transform=ax_axi.transAxes, color="white", fontsize=3.5, ha="center", va="top")
                        ax_axi.text(0.5, 0.02, "P", transform=ax_axi.transAxes, color="white", fontsize=3.5, ha="center", va="bottom")

            # ---- Single colorbar (both conditions share the same colormap), placed once
            # ---- in figure-fraction coordinates (independent of the participant grid) ----
            ax_cbar = fig.add_axes([0.93, 0.65, 0.025, 0.15])
            norm = plt.Normalize(vmin=stat_min, vmax=stat_max)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])

            cbar = fig.colorbar(sm, cax=ax_cbar)
            cbar.ax.set_yticks([])
            cbar.ax.set_frame_on(False)

            # Label sits above the colorbar (horizontal, not rotated) so it can't collide
            # with the last participant's title, regardless of figure width/subject count.
            cbar.ax.text(0.8, 1.3, "t-score\n(uncorrected)", fontsize=7, va="bottom", ha="center", transform=cbar.ax.transAxes)
            cbar.ax.text(0.9, -0.1, f"{stat_min:.1f}", fontsize=7,va="center", ha="right", transform=cbar.ax.transAxes)
            cbar.ax.text(0.9, 1.1, f"{stat_max:.1f}", fontsize=7, va="center", ha="right", transform=cbar.ax.transAxes)

            # --- Save figure ---
            fig.savefig(output_fname, dpi=600)
            plt.close(fig)
        
        else:
            print("First level figure already exists, put redo=True to regenerate the figure")
        
        return output_fname
    
    def plot_fmri_maps(self, i_fnames=None, output_fname=None, stat_min=2.3, stat_max=5,titles = ["shimBase", "shimSlice"],
                  background_fname=None, cbar_label='t-value', cmap="autumn",
                  mask_fname=None, underlay_fname=None, task_name=None, verbose=True, redo=False):

        if output_fname is None:
            raise ValueError("output_dir is empty")
        if i_fnames is None or len(i_fnames) == 0:
            raise ValueError("i_fnames_pair is empty")
        if background_fname is None:
            raise ValueError("Please provide PAM50 template filename")

        assert len(i_fnames) in [1, 2], f"Expected 1 or 2 maps, got {len(i_fnames)}"
        n_maps = len(i_fnames)

        # --- Figure and gridspec ---
        if not os.path.exists(output_fname) or redo:
            fig = plt.figure(figsize=(n_maps, 3))  # width scales with number of maps
            fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.1)

            height_ratios = [6, 0.2]
            gs = fig.add_gridspec(nrows=2, ncols=n_maps,
                                height_ratios=height_ratios,
                                width_ratios=[1] * n_maps,
                                hspace=0.01, wspace=0.05)

            # --- Load template, mask, and underlay ---
            template_img = nib.load(background_fname)
            template_data = nib.as_closest_canonical(template_img).get_fdata()

            if underlay_fname is not None:
                underlay_data = nib.as_closest_canonical(nib.load(underlay_fname)).get_fdata()

            # --- Plotting ---
            num_voxels_list = []
            values_list = []

            for i, fname in enumerate(i_fnames):
                stat_img = nib.as_closest_canonical(nib.load(fname))
                statmap_data = stat_img.get_fdata()

                num_voxels_list.append(np.nansum(statmap_data > stat_min))
                values_list.append(statmap_data.flatten())

                # --- Coronal (max-intensity projection across the AP axis) ---
                # A single fixed y-slice can miss the true peak entirely if it doesn't
                # happen to sit at that exact AP coordinate; MIP across the full AP
                # extent (matching plot_first_level_maps) shows any suprathreshold signal
                # regardless of where along that axis it falls.
                x_min, x_max = 35, 105
                # See plot_first_level_maps for the PAM50-cropping caveat: this indexes
                # template/PAM50_t2.nii.gz's 441-slice (0.5mm) crop, not the full 982-slice
                # PAM50 template. z=175-333 here is PAM50 z=-199.3mm to -120.3mm.
                z_min, z_max = 175, 333
                y_slice = 72  # anatomical background slice only
                cor_slice = np.max(statmap_data[x_min:x_max, :, z_min:z_max], axis=1)
                cor_slice = np.where(cor_slice > stat_min, cor_slice, np.nan)
                cor_slice = cor_slice.T

                ax_cor = fig.add_subplot(gs[0, i])
                template_cor = template_data[x_min:x_max, y_slice, z_min:z_max].T
                ax_cor.imshow(template_cor, cmap="gray", origin="lower", aspect="auto")

                # if there are only nan or 0 values, skip plotting the statmap to avoid showing a blank colorbar
                if np.nansum(cor_slice) == 0:
                    print(f"warning: no suprathreshold voxels found for {titles[i]} (MIP across AP axis), skipping statmap overlay")
                else:
                    im_cor = ax_cor.imshow(cor_slice, cmap=cmap, origin="lower", vmin=stat_min, vmax=stat_max, aspect="auto")

                ax_cor.axis("off")
                ax_cor.set_title(titles[i], color="black", fontweight='bold', fontsize=9, fontname="Arial")

                # orientation labels only on first map
                if i == 0:
                    ax_cor.text(0.05, 0.05, "L", transform=ax_cor.transAxes, color="white", fontsize=7, ha="left", va="bottom")
                    ax_cor.text(0.95, 0.05, "R", transform=ax_cor.transAxes, color="white", fontsize=7, ha="right", va="bottom")
                    
            # -- Shared colorbar
            cbar = self.plot_colorbar(
                fig=fig,
                stat_min=stat_min,
                stat_max=stat_max,
                cmap=cmap,
                label=cbar_label,
                left=0.4 if n_maps == 2 else 0.35 ,
                bottom=0.08, 
                width=0.35 if n_maps == 2 else 0.44, 
                height=0.02
                )

            plt.savefig(output_fname, transparent=True, dpi=600)
            plt.close(fig)

        return output_fname
    
    def plot_fmri_maps_axial(self, i_fnames=None, output_fname=None, stat_min=2.3, stat_max=5,
                          titles=["shimBase", "shimSlice"], background_fname=None, cbar_label='t-value', cmap="autumn",
                          z_slices=None, n_slices=6, mip=False, mask_fname=None, underlay_fname=None,
                          task_name=None, verbose=True, redo=False):
        """
        mip=True collapses the axial view to a single row per map, showing the
        max-intensity projection across the full S-I (z) axis instead of a grid of
        discrete slices -- avoids missing signal that doesn't happen to fall on one
        of the selected z levels (same rationale as the coronal MIP used elsewhere here).
        """

        if output_fname is None:
            raise ValueError("output_fname is empty")
        if i_fnames is None or len(i_fnames) == 0:
            raise ValueError("i_fnames is empty")
        if background_fname is None:
            raise ValueError("Please provide PAM50 template filename")

        assert len(i_fnames) in [1, 2], f"Expected 1 or 2 maps, got {len(i_fnames)}"
        n_maps = len(i_fnames)
        if titles is None:
            titles = [f"map{i}" for i in range(n_maps)]

        n_rows_plot = 1 if mip else n_slices

        if not os.path.exists(output_fname) or redo:
            fig = plt.figure(figsize=(n_maps * 0.6, n_rows_plot * 0.6))
            fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.05)

            gs = fig.add_gridspec(nrows=n_rows_plot, ncols=n_maps,
                                hspace=0.01, wspace=0.05)

            # --- Load template ---
            template_img = nib.load(background_fname)
            template_data = nib.as_closest_canonical(template_img).get_fdata()

            if underlay_fname is not None:
                underlay_data = nib.as_closest_canonical(nib.load(underlay_fname)).get_fdata()

            # --- Crop window ---
            crop_x, crop_y = 30, 30

            # --- Determine z slices from first map (multi-slice mode only) ---
            stat_img0 = nib.as_closest_canonical(nib.load(i_fnames[0]))
            statmap_data0 = stat_img0.get_fdata()
            x0 = statmap_data0.shape[0] // 2
            y0 = statmap_data0.shape[1] // 2
            x_min_axi, x_max_axi = x0 - crop_x, x0 + crop_x
            y_min_axi, y_max_axi = y0 - crop_y, y0 + crop_y
            crop_stat0 = statmap_data0[x_min_axi:x_max_axi, y_min_axi:y_max_axi, :]

            if not mip:
                if z_slices is not None and len(z_slices) == n_slices:
                    selected_z = z_slices
                else:
                    active_z = np.where(np.nanmax(crop_stat0, axis=(0, 1)) > stat_min)[0]
                    if len(active_z) >= n_slices:
                        indices = np.linspace(0, len(active_z) - 1, n_slices, dtype=int)
                        selected_z = active_z[indices]
                    else:
                        selected_z = np.linspace(0, crop_stat0.shape[2] - 1, n_slices, dtype=int)

            # --- Plot: col = map, row = slice (or single MIP row) ---
            for col, fname in enumerate(i_fnames):
                stat_img = nib.as_closest_canonical(nib.load(fname))
                statmap_data = stat_img.get_fdata()
                crop_stat = statmap_data[x_min_axi:x_max_axi, y_min_axi:y_max_axi, :]
                crop_tmpl = template_data[x_min_axi:x_max_axi, y_min_axi:y_max_axi, :]

                if mip:
                    z_mid = crop_tmpl.shape[2] // 2  # anatomical background only, not the overlay
                    ax = fig.add_subplot(gs[0, col])

                    tmpl_slice = crop_tmpl[:, :, z_mid].T
                    ax.imshow(tmpl_slice, cmap="gray", origin="lower", aspect="equal")

                    if underlay_fname is not None:
                        underlay_slice = underlay_data[x_min_axi:x_max_axi, y_min_axi:y_max_axi, z_mid].T
                        ax.imshow(underlay_slice, cmap="gray", origin="lower", aspect="auto", alpha=0.1)

                    stat_mip = np.max(crop_stat, axis=2)
                    stat_mip = np.where(stat_mip > stat_min, stat_mip, np.nan).T

                    if np.nansum(stat_mip) > 0:
                        ax.imshow(stat_mip, cmap=cmap, origin="lower",
                                vmin=stat_min, vmax=stat_max, aspect="auto")

                    ax.axis("off")

                    ax.set_title(titles[col], color="black", fontweight='bold',
                                fontsize=9, fontname="Arial")
                    continue

                for row, z in enumerate(selected_z):
                    ax = fig.add_subplot(gs[row, col])

                    # Background
                    tmpl_slice = crop_tmpl[:, :, z].T
                    ax.imshow(tmpl_slice, cmap="gray", origin="lower", aspect="equal")

                    if underlay_fname is not None:
                        underlay_slice = underlay_data[x_min_axi:x_max_axi, y_min_axi:y_max_axi, z].T
                        ax.imshow(underlay_slice, cmap="gray", origin="lower", aspect="auto", alpha=0.1)

                    # Stat overlay
                    stat_slice = crop_stat[:, :, z].copy()
                    stat_slice = np.where(stat_slice > stat_min, stat_slice, np.nan)
                    stat_slice = stat_slice.T

                    if np.nansum(stat_slice) > 0:
                        ax.imshow(stat_slice, cmap=cmap, origin="lower",
                                vmin=stat_min, vmax=stat_max, aspect="auto")

                    ax.text(0.5, 0.01, f"z={z}", color="white", fontsize=5,
                            ha="center", va="bottom", transform=ax.transAxes)
                    ax.axis("off")

                    # Orientation labels on first slice of first col only
                    if row == 0 and col == 0:
                        ax.text(0.02, 0.5, "L", transform=ax.transAxes, color="white", fontsize=7, ha="left", va="center")
                        ax.text(0.98, 0.5, "R", transform=ax.transAxes, color="white", fontsize=7, ha="right", va="center")
                        ax.text(0.5, 0.95, "A", transform=ax.transAxes, color="white", fontsize=7, ha="center", va="top")
                        #ax.text(0.5, 0.05, "P", transform=ax.transAxes, color="white", fontsize=7, ha="center", va="bottom")

                    # Column title on first row only
                    if row == 0:
                        ax.set_title(titles[col], color="black", fontweight='bold',
                                    fontsize=9, fontname="Arial")

            plt.savefig(output_fname, transparent=True, dpi=600)
            plt.close(fig)

        return output_fname

    def plot_colorbar(self, fig, stat_min, stat_max, cmap='autumn',
                  left=0.03, bottom=0.05, width=0.15, height=0.04,
                  label='t-value', fontsize=7.5):
        """
        Plot a shared colorbar on a figure.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
        stat_min : float
            Minimum value of the colorbar
        stat_max : float
            Maximum value of the colorbar
        cmap : str
            Colormap name (default: 'autumn')
        left : float
            Left position of the colorbar axes (default: 0.03)
        bottom : float
            Bottom position of the colorbar axes (default: 0.05)
        width : float
            Width of the colorbar axes (default: 0.02)
        height : float
            Height of the colorbar axes (default: 0.15)
        label : str
            Label of the colorbar (default: 't-score')
        fontsize : int
            Font size for label and tick text (default: 6)

        Returns
        -------
        cbar : matplotlib.colorbar.Colorbar
        """

        cbar_ax = fig.add_axes([left, bottom, width, height])
        norm = plt.Normalize(vmin=stat_min, vmax=stat_max)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_label(label, fontsize=fontsize, labelpad=3, fontweight='bold', fontname="Arial")

        cbar.ax.text(1.4, 0.3, f"{stat_max:.1f}", fontsize=fontsize, va='center', ha='right',
                    color='black', transform=cbar.ax.transAxes)
        cbar.ax.text(-0.1, 0.3, f"{stat_min:.1f}", fontsize=fontsize, va='center', ha='right',
                    color='black', transform=cbar.ax.transAxes)
        cbar.ax.set_xticks([])
        cbar.ax.set_frame_on(False)

        return cbar
    
    def bar_plot(self,csv_pair=None,metric="nonzero_voxels",output_fname=None, colors = None, maps_name=None, figsize=(1.8, 2.5),width=0.5, alpha=0.8,redo=False):
        """
        Plot a bar chart of metrics loaded from a pair of CSV files.

        Parameters
        ----------
        csv_pair : list
            List of two CSV filenames (output of extract_metrics)
        output_fname : str, optional
            Path to save the figure. If None, figure is returned without saving.
        colors : list, optional
            Bar colors (default: ["#43BA8C", "#F5AD27"])
        maps_name : list, optional
            X-tick labels (default: ["shimBase", "shimSlice"])
        figsize : tuple
            Figure size (default: (1.5, 2))
        width : float
            Bar width (default: 0.5)
        alpha : float
            Bar transparency (default: 0.7)
        metric : str
            Column name to plot from the CSV (default: "nonzero_voxels")

        """
        if not os.path.exists(output_fname) or redo:
            if csv_pair is None:
                raise ValueError("Please provide a list of two CSV filenames.")
        
            if colors is None:
                colors = ["#ADA8A8","#ED263F"]
            if maps_name is None:
                maps_name = ["shimBase", "shimSlice"]

            # --- Load metric from each CSV ---
            values = [pd.read_csv(f)[metric].values[0] for f in csv_pair]

            # --- Plot ---
            fig, ax = plt.subplots(figsize=figsize)
            fig.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.25)

            ax.bar(range(len(values)), values, color=colors, width=0.5, alpha=alpha)
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(
                [maps_name[i] for i in range(len(values))],
                rotation=45, fontsize=11, fontweight='bold', fontname="Arial", ha='right')
            ax.set_ylabel("# significant voxels", fontsize=12, fontweight='bold', fontname="Arial")
            ax.tick_params(axis='y', labelsize=7)
            #ax.yaxis.set_label_coords(-0.9, 0.5)
            ax.tick_params(axis='y', which='both', pad=2)
            ax.spines['left'].set_position(('outward', 10))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            plt.tight_layout()

            plt.savefig(output_fname,transparent=True, dpi=600)
            plt.close(fig)
        
        return output_fname

    def plot_dist(self, csv_pair=None,output_fname=None,colors=None, maps_name=None,bins=100,figsize=(1.8, 2.3),width=0.5, alpha=0.8, redo=False):
        """
        Plot a bar chart of suprathreshold voxel counts as a standalone figure.

        Parameters
        ----------
        csv_pair : list
            List of two CSV filenames 
        output_fname : str, optional
            Path to save the figure. If None, figure is returned without saving.
        colors : list, optional
            Bar colors (default: ["#43BA8C", "#F5AD27"])
        maps_name : list, optional
            X-tick labels (default: ["baseShim", "SliceShim"])
        figsize : tuple
            Figure size (default: (1.5, 2))
        width : float
            Bar width (default: 0.5)
        alpha : float
            Bar transparency (default: 0.7)

        Returns
        -------
        output filename
        """

        if not os.path.exists(output_fname) or redo:
            if csv_pair is None:
                raise ValueError("Please provide a list of two CSV filenames.")
        
            if colors is None:
                colors = ["#ADA8A8","#D61532"]
            if maps_name is None:
                maps_name = ["shimBase", "shimSlice"]

            values_list = [pd.read_csv(f)["voxels_values"].values for f in csv_pair]
            # --- Plot ---
            fig, ax = plt.subplots(figsize=figsize)

            for i, values in enumerate(values_list):
                if i==1:
                    alpha=0.9
                values_clean = values[values != 0]
                ax.hist(values_clean, bins=bins, color=colors[i], alpha=alpha,
                        label=maps_name[i], density=False)

            ax.set_xlabel("t-value", fontsize=12, fontweight='bold', fontname="Arial")
            ax.set_ylabel("# significant voxels", fontsize=12, fontweight='bold', fontname="Arial")
            ax.tick_params(axis='both', labelsize=7)

            # Get current handles and labels
            handles, labels = ax.get_legend_handles_labels()
            if len(labels) == 2 and "shimBase" in labels and "shimSlice" in labels:
                # Make sure the order is shimBase, shimSlice
                handles = [handles[labels.index("shimBase")], handles[labels.index("shimSlice")]]
                labels = ["shimBase", "shimSlice"]
                ax.legend(handles, labels, fontsize=8.5, frameon=False, loc='upper right')
            else:
                ax.legend(fontsize=8.5, frameon=False, loc='upper right')

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            plt.tight_layout()

            plt.savefig(output_fname,transparent=True, dpi=600)
            plt.close(fig)
        
        return output_fname

    def boxplots(self, csv_file=None,df=None,output_fname=None, stats_file=None,x_data=None, x_order=None, y_data=None, hue=None, hue_order=None,specify_y_label=None,output_dir=None, color=None, indiv_values=False,indiv_hue=None, indiv_color=None, plot_legend=True, output_tag='', ymin=6, ymax=17,height=2.5,aspect=0.6, invers_axes=False,indiv=False, group=False, show_pvalues_if_sig=True,plot_xlabels=True, redo=False):
        """
        Create matrix of correlation boxplots with matching box outline and whisker colors.
        """

        if not os.path.exists(output_fname) or redo:
            if csv_file:
                df = pd.read_csv(csv_file)
            

            # Set style and default palette if not provided
            if color is None:
                color = ["#ADA8A8","#ED263F"]
            if hue is None:
                hue = x_data
                hue_order = x_order
            
            if invers_axes:
                x_data_f=y_data
                y_data_f=x_data
            else:
                x_data_f=x_data
                y_data_f=y_data


            #--- Create the boxplot
            g = sns.catplot(
                    x=x_data_f, 
                    y=y_data_f, 
                    data=df,  
                    kind="box",  
                    linewidth=2, 
                    #color=color,  # Use the provided palette
                    medianprops=dict(color="white",alpha=0.5),  # Set median line color to white
                    #hue=None,
                    order=x_order, 
                    #hue_order=None,
                    fliersize=0,  # Remove outliers' markers
                    height=height,
                    aspect=aspect,
                    legend=plot_legend
                )
            fig = g.figure
            current_width = fig.get_figwidth()
            #fig.set_size_inches(current_width, 1)  # keep width, reduce height

            # Apply custom outline and whisker colors to match the palette
            for ax in g.axes.flat:
                # Add a horizontal line at y=0
                if invers_axes:
                    ax.axvline(0, color='grey', linestyle='--', linewidth=1)
                else:
                    ax.axhline(0, color='grey', linestyle='--', linewidth=1)
                
                # Change whisker colors
                for i, box in enumerate(ax.patches):  # Access the box patches
                    category = df[x_data_f].unique()[i % len(df[x_data_f].unique())]  # Use modulus to loop over categories
                    color_index = list(df[x_data_f].unique()).index(category)  # Get the index of the category in the unique list
                    
                    # Set the box color and alpha
                    box.set_color(color[color_index])  # Set box color
                    box.set_alpha(0.4)  # Set alpha for the box
                    
                    whisker_lines = ax.lines[i * 6:i * 6 + 2]  # Whiskers are the first two lines for each box
                    for whisker in whisker_lines:
                        whisker.set_color(color[color_index])  # Set the whisker color
                        whisker.set_alpha(0.4)  # Set alpha for whiskers

                    cap_lines = ax.lines[i * 6 + 2:i * 6 + 4]  # Caps are the next two lines for each box
                    for cap in cap_lines:
                        cap.set_color(color[color_index])  # Set the cap color
                        cap.set_alpha(0.4)


                    # Loop through each box and set outline color
                    # Get the current category for the box
                    category = df[x_data_f].unique()[i % len(df[x_data_f].unique())]  # Use modulus to loop over categories
                    color_index = list(df[x_data_f].unique()).index(category)  # Get the index of the category in the unique list

                    # Set the box color
                    box.set_color(color[color_index])  # Set box color
                    
                    # Get the bounding box by extracting the vertices of the path
                    vertices = box.get_path().vertices
                    x_pos = vertices[:, 0].min()  # Minimum x value
                    y_pos = vertices[:, 1].min()  # Minimum y value
                    box_width = vertices[:, 0].max() - x_pos  # Width
                    box_height = vertices[:, 1].max() - y_pos  # Height

                    # Create a new outline with lower alpha for the edge
                    outline = plt.Rectangle(
                        (x_pos, y_pos),  # Position as a tuple
                        box_width,  # Width
                        box_height,  # Height
                        fill=False,  # No fill for the outline
                        edgecolor=color[color_index],  # Same color as the box
                        lw=0,  # Line width
                        alpha=0.4  # Set alpha for transparency of the outline
                    )
                    ax.add_patch(outline)  # Add the outline to the axis

            # ------- Add individual points if requested
            if indiv_values:
                sns.stripplot(
                    x=x_data_f, 
                    y=y_data_f, 
                    data=df, 
                    hue=hue, 
                    hue_order=hue_order,
                    size=5,
                    palette=indiv_color if indiv_color else color,
                    #palette=palette, 
                    linewidth=0, 
                    alpha=0.7,
                    edgecolor='white',
                    jitter=False #set 0.25 to add jitter between individual points
                )

                # Draw lines between points from the same individual
                ax = g.axes.flat[0]

                x_positions = {}
                collections = [c for c in ax.collections if isinstance(c, plt.matplotlib.collections.PathCollection)]  # Get the jittered x positions from the stripplot collections

                for coll_idx, collection in enumerate(collections):
                    offsets = collection.get_offsets()
                    category = x_order[coll_idx] if x_order else df[x_data_f].unique()[coll_idx]
                    for x_pos, y_pos in offsets:
                        # Match y value back to the ID
                        matched = df[(df[x_data_f] == category) & (np.isclose(df[y_data_f], y_pos))]
                        if not matched.empty:
                            ind_id = matched.iloc[0]['IDs']
                            if ind_id not in x_positions:
                                x_positions[ind_id] = {}
                            x_positions[ind_id][category] = (x_pos, y_pos)

                # Draw lines using the recovered jittered positions
                for ind_id, coords in x_positions.items():
                    ordered_cats = [c for c in (x_order if x_order else df[x_data_f].unique()) if c in coords]
                    xs = [coords[c][0] for c in ordered_cats]
                    ys = [coords[c][1] for c in ordered_cats]
                    ax.plot(xs, ys, color='grey', alpha=1,linestyle='--', linewidth=1, zorder=1) #linestyle='--',
            
            # ------- Add significance annotation if stats_file provided
            if stats_file is not None:
                stats_df = pd.read_csv(stats_file)
                ax = g.axes.flat[0]

                # Get x positions of the two conditions
                if x_order:
                    x1 = x_order.index(stats_df['cond1'].values[0])
                    x2 = x_order.index(stats_df['cond2'].values[0])
                else:
                    cats = list(df[x_data_f].unique())
                    x1 = cats.index(stats_df['cond1'].values[0])
                    x2 = cats.index(stats_df['cond2'].values[0])

                stars = stats_df['significance'].values[0]
                if stars != 'ns' and show_pvalues_if_sig:
                    pvalue = stats_df['p_value'].values[0]
                    # sig_annotation = f"{stars}\n(p={pvalue:.3f})"
                    sig_annotation = f"p={pvalue:.3f}"
                else:
                    sig_annotation = stars

                # Draw bracket
                y_bracket = ymax * 0.97  # just below the top
                y_tip     = y_bracket - (ymax - ymin) * 0.02
                bracket_color = 'black'

                ax.plot([x1, x1, x2, x2], 
                        [y_tip, y_bracket, y_bracket, y_tip],
                        color=bracket_color, linewidth=1)
                ax.text((x1 + x2) / 2, y_bracket + (ymax - ymin) * 0.01,
                        sig_annotation,fontname="Arial",
                        ha='center', va='bottom',
                        fontsize=8, color=bracket_color)
            
            ax.set_xlabel('')
            y_label=specify_y_label if specify_y_label else y_data

            ax.set_ylabel(y_label, fontsize=12, fontname="Arial",fontweight='bold')
            ax.tick_params(axis='y', labelsize=8)
            

            if output_tag:
                g.set(title=output_tag)

            if invers_axes:
                g.set(xlim=(ymin, ymax))
            else:
                g.set(ylim=(ymin, ymax))
            sns.despine(offset=5, trim=True)
            if plot_legend:
                g.add_legend()
            else:
                plt.legend([],[], frameon=False)
            
            ax.set_xticks(range(len(df[x_data_f].unique())))
            #if plot_xlabels==True:
            ax.set_xticklabels(x_order if x_order else df[x_data_f].unique(), 
                    rotation=45, fontsize=10, fontweight='bold', fontname="Arial", ha='right')
            #else:
             #   ax.set_xticklabels(['' ] * len(df[x_data_f].unique()))
            
            # Save the figure if requested
            plt.tight_layout(pad=0.1)
            plt.savefig(output_fname, dpi=600, transparent=True)
            plt.close()
        
        return output_fname

        
    def combine_plots(self, output_fname, map_files, graph_files=None,
                  axial_files=None,
                  map_titles=None, axial_titles=None, graph_titles=None,
                  label_idx=True,
                  figsize=(3.5, 3.5), graph_width_scale=1.0, graph_height_scale=1.0,
                  graph_col_scale=0.6, axial_col_scale=1.1, redo=False):

        n_maps = len(map_files)
        n_graphs = len(graph_files) if graph_files else 0
        n_axial = len(axial_files) if axial_files else 0

        assert n_maps in (1, 2), f"Expected 1 or 2 map_files, got {n_maps}"
        assert n_graphs in (0, 2, 4), f"Expected 0, 2 or 4 graph_files, got {n_graphs}"
        if axial_files:
            assert n_axial in (1, 2), f"Expected 1 or 2 axial_files, got {n_axial}"

        if not os.path.exists(output_fname) or redo:
            n_graph_cols = n_graphs // 2 if n_graphs > 0 else 0
            n_axial_cols = n_axial if axial_files else 0
            n_rows = 2

            # Read image sizes
            map_img = mpimg.imread(map_files[0])
            map_h, map_w = map_img.shape[:2]
            map_col_width = figsize[1] * (map_w / map_h)

            graph_cell_height = figsize[1] / n_rows

            if graph_files:
                graph_img = mpimg.imread(graph_files[0])
                graph_h, graph_w = graph_img.shape[:2]
                graph_col_width = graph_cell_height * (graph_w / graph_h)
                graph_widths = [(graph_col_width / map_col_width) * graph_col_scale] * n_graph_cols
            else:
                graph_widths = []

            # Width ratios
            map_widths = [0.6] * n_maps

            axial_widths = []
            if axial_files:
                axial_img = mpimg.imread(axial_files[0])
                axial_h, axial_w = axial_img.shape[:2]
                axial_col_width = graph_cell_height * (axial_w / axial_h)
                axial_widths = [(axial_col_width / map_col_width) * axial_col_scale] * n_axial_cols

            if graph_files:
                graph_widths = [(graph_col_width / map_col_width) * graph_col_scale] * n_graph_cols
            else:
                graph_widths = []
            width_ratios = map_widths + axial_widths + graph_widths

            total_width = sum(map_widths) + sum(axial_widths) + sum(graph_widths)

            new_figwidth = min(figsize[1] * (map_col_width * sum(width_ratios) / sum(map_widths)), figsize[0])
            fig = plt.figure(figsize=(new_figwidth, figsize[1]))

            n_cols = n_maps + n_axial_cols + n_graph_cols
            gs = fig.add_gridspec(n_rows, n_cols,
                                width_ratios=width_ratios,
                                hspace=0.05, wspace=0.05)

            label_y = 0.98
            label_idx = 0 if label_idx else None

            # --- Map columns: span both rows ---
            
            for i, fname in enumerate(map_files):
                ax = fig.add_subplot(gs[:, i])
                img = mpimg.imread(fname)
                h, w = img.shape[:2]
                ax.imshow(img, aspect='auto')
                ax.set_xlim(0, w)
                ax.set_ylim(h, 0)
                ax.axis('off')
                
                if label_idx:
                    ax.text(0.0, 1.0, f"{chr(65 + label_idx)}.",
                    ha='left', va='bottom',
                    fontsize=8, fontweight='bold', fontname="Arial",
                    transform=ax.transAxes, clip_on=False)
                    label_idx += 1
                if map_titles and i < len(map_titles):
                    ax.text(0.5, label_y, map_titles[i],
                            ha='center', va='bottom',
                            fontsize=12, fontweight='bold', fontname="Arial",
                            transform=ax.transAxes, clip_on=False)

            # --- Axial columns: span both rows ---
            if axial_files:
                for i, fname in enumerate(axial_files):
                    col = n_maps + i
                    ax = fig.add_subplot(gs[:, col])
                    img = mpimg.imread(fname)
                    h, w = img.shape[:2]
                    ax.imshow(img, aspect='auto')
                    ax.set_xlim(0, w)
                    ax.set_ylim(h, 0)
                    ax.axis('off')
                    if label_idx:
                        ax.text(0.0, 1.0, f"{chr(65 + label_idx)}.",
                            ha='left', va='bottom',
                            fontsize=8, fontweight='bold', fontname="Arial",
                            transform=ax.transAxes, clip_on=False)
                        label_idx += 1
                    if axial_titles and i < len(axial_titles):
                        ax.text(0.5, 1.0, axial_titles[i],
                                ha='center', va='bottom',
                                fontsize=12, fontweight='bold', fontname="Arial",
                                transform=ax.transAxes, clip_on=False)

            # --- Graph columns: 2-row grid ---
            if graph_files:
                for i, fname in enumerate(graph_files):
                    row = i // n_graph_cols
                    col = n_maps + n_axial_cols + (i % n_graph_cols)
                    ax = fig.add_subplot(gs[row, col])
                    img = mpimg.imread(fname)

                    margin_x = (1 - graph_width_scale) / 2
                    margin_y = (1 - graph_height_scale) / 2
                    ax_inner = ax.inset_axes([margin_x, margin_y, graph_width_scale, graph_height_scale])
                    ax_inner.imshow(img, aspect='auto')
                    ax_inner.axis('off')
                    ax.axis('off')

                    if graph_titles and i < len(graph_titles):
                        ax_inner.set_title(graph_titles[i], fontsize=7,
                                        fontweight='bold', fontname="Arial")
                    
                    if label_idx:
                        ax.text(0.0, 1.0, f"{chr(65 + label_idx)}.",
                            ha='left', va='bottom',
                            fontsize=8, fontweight='bold', fontname="Arial",
                            transform=ax.transAxes, clip_on=False)
                        label_idx += 1

            fig.subplots_adjust(wspace=0.05, hspace=0.05,
                                left=0.01, right=0.99, top=0.93, bottom=0.01)
            plt.savefig(output_fname, dpi=600, transparent=True, bbox_inches='tight')
            plt.close()

