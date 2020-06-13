# Import everything from the commons module.
# In the .pyx file, Cython declared variables will also get cimported.
from commons import *

# Cython imports
cimport('from interactions import particle_particle')



# Function implementing pairwise collision
@cython.header(
    # Arguments
    interaction_name=str,
    receiver='Component',
    supplier='Component',
    ᔑdt_rungs=dict,
    rank_supplier='int',
    only_supply='bint',
    pairing_level=str,
    tile_indices_receiver='Py_ssize_t[::1]',
    tile_indices_supplier_paired='Py_ssize_t**',
    tile_indices_supplier_paired_N='Py_ssize_t*',
    extra_args=dict,
    # Locals
    apply_to_i='bint',
    apply_to_j='bint',
    probability='double',
    w='double',
    n='double',
    vel_rel_dependence='double',
    angle_cutoff='double',
    rung_index_i='signed char',
    rung_index_j='signed char',
    rung_index_max='signed char',
    collision_factors='double[::1]',
    collision_factors_ptr='double*',
    d2='double',
    collision_length2='double',
    vel_relative='double',
    vel_dependence=str,
    n_interactions='Py_ssize_t',
    distribution=str,
    subtiling_r='Tiling',
    v_to_w='double',
    v_to_w2='double',
    momx_i='double',
    momy_i='double',
    momz_i='double',
    returns='void',
)
def collision_pairwise(
    interaction_name, receiver, supplier, ᔑdt_rungs, rank_supplier, only_supply, pairing_level,
    tile_indices_receiver, tile_indices_supplier_paired, tile_indices_supplier_paired_N,
    extra_args,
):
    # Extract momentum buffers
    momx_r = receiver.momx
    momy_r = receiver.momy
    momz_r = receiver.momz
    momx_s = supplier.momx
    momy_s = supplier.momy
    momz_s = supplier.momz
    Δmomx_s = supplier.Δmomx
    Δmomy_s = supplier.Δmomy
    Δmomz_s = supplier.Δmomz
    # Masses
    mass_r = receiver.mass
    mass_s = supplier.mass
    # The sum of squared collision lengths
    collision_length_r = get_shortrange_param(receiver, 'collision', 'scale')
    collision_length_s = get_shortrange_param(supplier, 'collision', 'scale')
    collision_length2 = collision_length_r**2 + collision_length_s**2
    collision_range2 = get_shortrange_param((receiver, supplier), 'collision', 'range')**2
    # Collision cross section per mass
    σ_per_mass = extra_args['σ_per_mass'][receiver.name, supplier.name]
    distribution = extra_args['distribution'][receiver.name, supplier.name]
    vel_dependence = extra_args['vel_dependence'][receiver.name, supplier.name]
    σ = σ_per_mass*(mass_s + mass_r)/2
    collision_factors = (ℝ[2*π]*collision_length2)**(-1.5)*σ*ᔑdt_rungs['a**(-1)']
    collision_factors_ptr = cython.address(collision_factors[:])
    # Velocity normilization factor
    w = extra_args['velocity_normalization_factor'][receiver.name, supplier.name]
    n = extra_args['vel_dependence_power'][receiver.name, supplier.name]
    # Angular distribution parameters
    angle_cutoff = 1000 #cutoff for forward / sideway distribution (so it doesnt explode)
    # Counter
    n_interactions = 0
    j = -1
    # Loop over all (receiver, supplier) particle pairs (i, j)
    for i, j, rung_index_i, rung_index_j, x_ji, y_ji, z_ji, apply_to_i, apply_to_j, particle_particle_t_begin, subtiling_r in particle_particle(
        receiver, supplier, pairing_level,
        tile_indices_receiver, tile_indices_supplier_paired, tile_indices_supplier_paired_N,
        rank_supplier, interaction_name, only_supply,
    ):
        # Translate coordinates so they correspond to the nearest image
        if x_ji > ℝ[0.5*boxsize]:
            x_ji -= boxsize
        elif x_ji < ℝ[-0.5*boxsize]:
            x_ji += boxsize
        if y_ji > ℝ[0.5*boxsize]:
            y_ji -= boxsize
        elif y_ji < ℝ[-0.5*boxsize]:
            y_ji += boxsize
        if z_ji > ℝ[0.5*boxsize]:
            z_ji -= boxsize
        elif z_ji < ℝ[-0.5*boxsize]:
            z_ji += boxsize
        d2 = x_ji**2 + y_ji**2 + z_ji**2
        if d2 > collision_range2:
             continue
        # Extract momenta
        momx_i     = momx_r[i]
        momy_i     = momy_r[i]
        momz_i     = momz_r[i]
        momx_j_ori = momx_s[j]
        momy_j_ori = momy_s[j]
        momz_j_ori = momz_s[j]
        vel_relative = sqrt(
            + (momx_i*ℝ[1/mass_r] - momx_j_ori*ℝ[1/mass_s])**2
            + (momy_i*ℝ[1/mass_r] - momy_j_ori*ℝ[1/mass_s])**2
            + (momz_i*ℝ[1/mass_r] - momz_j_ori*ℝ[1/mass_s])**2
        )
        rung_index_max = (rung_index_i if rung_index_i > rung_index_j else rung_index_j)
        # Compute collision probability
        v_to_w = vel_relative/w
        v_to_w2 = v_to_w**2
        if 𝔹[vel_dependence == 'independent']:
            vel_rel_dependence = 1
        elif 𝔹[vel_dependence == 'yukawa']:
            vel_rel_dependence = v_to_w2/(1 + v_to_w2)
        elif 𝔹[vel_dependence == 'power']:
            vel_rel_dependence = v_to_w**n
        elif 𝔹[vel_dependence == 'funky']:
            vel_rel_dependence = 1/((1 - exp(-v_to_w))*v_to_w2)
        else:
            vel_rel_dependence = 0
        probability = collision_factors_ptr[rung_index_max]*exp(d2*ℝ[-1/(2*collision_length2)])*vel_relative*vel_rel_dependence
        # Determine whether collision takes place
        if random() > probability:
            continue
        # Collision takes place!
        n_interactions += 1
        # Draw collision angles
        ϕ = ℝ[2*π]*random()
        with unswitch:
            if distribution == 'isotropic':
                cosθ = 2*random() - 1
            elif distribution == 'forward':
                cosθ = 1 - 1/(angle_cutoff*random() + 1)
            else:
                cosθ = 1  # To satisfy the compiler
                abort(
                    f'Collision distribution "{distribution}" '
                    f'not implemented in collision_pairwise()'
                )
        # Boost to CM
        boostx = (momx_i + momx_j_ori)*ℝ[mass_r/(mass_r + mass_s)]
        boosty = (momy_i + momy_j_ori)*ℝ[mass_r/(mass_r + mass_s)]
        boostz = (momz_i + momz_j_ori)*ℝ[mass_r/(mass_r + mass_s)]
        momx_i -= boostx
        momy_i -= boosty
        momz_i -= boostz
        # Construct final momentum i in CM frame
        p_xy2inv = 1/(ℝ[momx_i**2 + momy_i**2] + machine_ϵ)
        p_xyz = sqrt(ℝ[momx_i**2 + momy_i**2] + momz_i**2)
        tmp = p_xy2inv*(momz_i - p_xyz)
        sinθ = sqrt(1-cosθ**2)
        sinϕ, cosϕ = sin(ϕ), cos(ϕ)
        momxy_i = momx_i*momy_i
        momx_i, momy_i, momz_i = (
            +sinθ*(sinϕ*ℝ[momxy_i*tmp]
                + cosϕ*(momz_i - momy_i**2*tmp)) + cosθ*momx_i,
            +sinθ*(cosϕ*ℝ[momxy_i*tmp]
                + sinϕ*(momz_i - momx_i**2*tmp)) + cosθ*momy_i,
            -sinθ*(cosϕ*momx_i + sinϕ*momy_i) + cosθ*momz_i,
        )
        # Get momentum j from momentum conservation
        momx_j = -momx_i
        momy_j = -momy_i
        momz_j = -momz_i
        # Boost back to lab
        momx_i += boostx
        momy_i += boosty
        momz_i += boostz
        momx_j += boostx*ℝ[mass_s/mass_r]
        momy_j += boosty*ℝ[mass_s/mass_r]
        momz_j += boostz*ℝ[mass_s/mass_r]
        # Apply momentum changes
        momx_r[i] = momx_i
        momy_r[i] = momy_i
        momz_r[i] = momz_i
        momx_s[j] = momx_j
        momy_s[j] = momy_j
        momz_s[j] = momz_j
        with unswitch:
            if rank_supplier != rank:
                Δmomx_s[j] += -momx_j_ori + momx_j
                Δmomy_s[j] += -momy_j_ori + momy_j
                Δmomz_s[j] += -momz_j_ori + momz_j
    # Add computation time to the running total,
    # for use with automatic subtiling refinement.
    if j != -1:
        particle_particle_t_final = time()
        subtiling_r.computation_time += particle_particle_t_final - particle_particle_t_begin
    # Add number of interactions to running total
    receiver.n_interactions[supplier.name] += n_interactions
