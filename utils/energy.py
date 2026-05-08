def calculate_ann_energy(total_macs):

    # 1 MAC ≈ 100 pJ
    return total_macs * 100.0


def calculate_snn_energy(total_spikes):

    # 1 SOP ≈ 1 pJ
    return total_spikes * 1.0
