rule preprocess:
    input:
        "data/X_dataset.biom"
        "data/Y_dataset.biom"
    output:
        "out-directory/scores.tsv"
        "out-directory/Lx.tsv"
        "out-directory/Ly.tsv"
    conda:
        "RoSCA.yaml"
    shell:
        r"""
        python preprocessing.py \
            --in {input} \
            --out {output}
        """

rule tune_rosca:
    input:
        X = "data/X_dataset.biom"
        Y = "data/Y_dataset.biom"
    output:
        "output/THDMI.summary.json"
    conda:
        "RoSCA.yaml"
    shell:
        r"""
        python auto_tuning.py \
            --X {input.X} \
            --Y {input.Y} \
            --out {output}
        """

rule plot_rosca:
    input:
        "tests/thdmi/output/THDMI.summary.json"
    output:
        "tests/thdmi/output/rosca_plots.pdf"
    conda:
        "RoSCA.yaml"
    shell:
        r"""
        jupyter nbconvert --to notebook --execute plotting.ipynb \
            --ExecutePreprocessor.timeout=-1 \
            --output plotting_executed.ipynb
        # then move/rename whatever plotting.ipynb writes to {output}
        """
