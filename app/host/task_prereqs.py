ImageDownload_prerequisites = {
    "Cutout download": "not processed"
}
GenerateThumbnail_prerequisites = {
    "Cutout download": "processed",
    "Generate thumbnail": "not processed",
}
MWEBV_Transient_prerequisites = {
    "Cutout download": "processed",
    "Transient MWEBV": "not processed",
}
HostMatch_prerequisites = {
    "Cutout download": "processed",
    "Transient MWEBV": "processed",
    "Host match": "not processed",
}
HostInformation_prerequisites = {
    "Cutout download": "processed",
    "Transient MWEBV": "processed",
    "Host match": "processed",
    "Host information": "not processed"
}
LocalAperturePhotometry_prerequisites = {
    "Cutout download": "processed",
    "Transient MWEBV": "processed",
    "Host match": "processed",
    "Host information": "processed",
    "Local aperture photometry": "not processed",
}
ValidateLocalPhotometry_prerequisites = {
    "Cutout download": "processed",
    "Transient MWEBV": "processed",
    "Host match": "processed",
    "Host information": "processed",
    "Local aperture photometry": "processed",
    "Validate local photometry": "not processed",
}
LocalHostSEDFitting_prerequisites = {
    "Cutout download": "processed",
    "Transient MWEBV": "processed",
    "Host match": "processed",
    "Host information": "processed",
    "Local aperture photometry": "processed",
    "Validate local photometry": "processed",
    "Local host SED inference": "not processed",
}
MWEBV_Host_prerequisites = {
    "Cutout download": "processed",
    "Transient MWEBV": "processed",
    "Host match": "processed",
    "Host information": "processed",
    "Host MWEBV": "not processed"
}
GlobalApertureConstruction_prerequisites = {
    "Cutout download": "processed",
    "Transient MWEBV": "processed",
    "Host match": "processed",
    "Host information": "processed",
    "Global aperture construction": "not processed",
}
GlobalAperturePhotometry_prerequisites = {
    "Cutout download": "processed",
    "Transient MWEBV": "processed",
    "Host match": "processed",
    "Host information": "processed",
    "Global aperture construction": "processed",
    "Global aperture photometry": "not processed",
}
ValidateGlobalPhotometry_prerequisites = {
    "Cutout download": "processed",
    "Transient MWEBV": "processed",
    "Host match": "processed",
    "Host information": "processed",
    "Global aperture construction": "processed",
    "Global aperture photometry": "processed",
    "Validate global photometry": "not processed",
}
GlobalHostSEDFitting_prerequisites = {
    "Cutout download": "processed",
    "Transient MWEBV": "processed",
    "Host match": "processed",
    "Host information": "processed",
    "Global aperture construction": "processed",
    "Global aperture photometry": "processed",
    "Validate global photometry": "processed",
    "Host MWEBV": "processed",
    "Global host SED inference": "not processed",
}
CropTransientImages_prerequisites = {
    "Cutout download": "processed",
    "Transient MWEBV": "processed",
    "Host match": "processed",
    "Host information": "processed",
    "Global aperture construction": "processed",
    "Global aperture photometry": "processed",
    "Validate global photometry": "processed",
    "Local aperture photometry": "processed",
    "Validate local photometry": "processed",
    "Crop transient images": "not processed",
}
GenerateThumbnailFinal_prerequisites = {
    "Cutout download": "processed",
    "Transient MWEBV": "processed",
    "Host match": "processed",
    "Host information": "processed",
    "Global aperture construction": "processed",
    "Global aperture photometry": "processed",
    "Validate global photometry": "processed",
    "Local aperture photometry": "processed",
    "Validate local photometry": "processed",
    "Generate thumbnail": "processed",
    "Crop transient images": "processed",
    "Generate thumbnail final": "not processed",
}
GenerateThumbnailSEDLocal_prerequisites = {
    "Cutout download": "processed",
    "Transient MWEBV": "processed",
    "Host match": "processed",
    "Host information": "processed",
    "Local aperture photometry": "processed",
    "Validate local photometry": "processed",
    "Local host SED inference": "processed",
    "Generate thumbnail SED local": "not processed",
}
GenerateThumbnailSEDGlobal_prerequisites = {
    "Cutout download": "processed",
    "Transient MWEBV": "processed",
    "Host match": "processed",
    "Host information": "processed",
    "Global aperture construction": "processed",
    "Global aperture photometry": "processed",
    "Validate global photometry": "processed",
    "Host MWEBV": "processed",
    "Global host SED inference": "processed",
    "Generate thumbnail SED global": "not processed",
}
