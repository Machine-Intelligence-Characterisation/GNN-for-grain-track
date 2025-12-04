import warnings


def pass_grain_property(source_graph, target_graph, grain_property_name):
    if grain_property_name not in source_graph.grain_properties:
        warnings.warn(f"Grain property {grain_property_name} not found in the current graph.")
        return

    if len(source_graph.grains) != len(target_graph.grains):
        raise ValueError("The number of grains in the source and target graphs must match.")

    for source_grain, target_grain in zip(source_graph.grains, target_graph.grains):
        if hasattr(source_grain, grain_property_name):
            property_value = getattr(source_grain, grain_property_name)
            target_grain.add_new_property(grain_property_name, property_value)
        else:
            warnings.warn(f"Grain {source_grain.grain_id} does not have property {grain_property_name}. Skipping.")

    target_graph.grain_properties.append(grain_property_name)


def pass_overlay_attr(source_graph, target_graph, attr_name):
    # For most of the cases, I guess the attr in source_graph won't be the same shape with target_graph. I feel this
    # function is important but need some refinement to make it work properly. Especially, some new function should
    # be added to the GrainGraph class to handle the overlay attributes properly.
    pass


    # if not hasattr(source_graph, attr_name):
    #     warnings.warn(f"Attribute {attr_name} not found in the source graph.")
    #     return
    #
    # if len(source_graph.grains) != len(target_graph.grains):
    #     raise ValueError("The number of grains in the source and target graphs must match.")
    #
    # attr_value = getattr(source_graph, attr_name)
    # setattr(target_graph, attr_name, attr_value)
    #
    # target_graph.overlay_attrs.append(attr_name)
    # warnings.warn(f"Attribute {attr_name} passed from source to target graph.")

