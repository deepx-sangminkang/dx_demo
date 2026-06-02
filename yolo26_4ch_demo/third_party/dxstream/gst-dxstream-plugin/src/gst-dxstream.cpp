#include "gst-dxinfer.hpp"
#include "gst-dxpostprocess.hpp"
#include "gst-dxpreprocess.hpp"
#include "gst-dxscale.hpp"
#include <gst/gst.h>

GST_DEBUG_CATEGORY(dxframemeta_cat);
GST_DEBUG_CATEGORY(dxobjectmeta_cat);
GST_DEBUG_CATEGORY(dxusermeta_cat);

static gboolean plugin_init(GstPlugin *plugin) {
    // debug category
    GST_DEBUG_CATEGORY_INIT(dxframemeta_cat, "dxframemeta", 0, "DX Frame Meta");
    GST_DEBUG_CATEGORY_INIT(dxobjectmeta_cat, "dxobjectmeta", 0, "DX Object Meta");
    GST_DEBUG_CATEGORY_INIT(dxusermeta_cat, "dxusermeta", 0, "DX User Meta");

    // Utility Elements
    if (!gst_element_register(plugin, "dxscale", GST_RANK_NONE,
                              GST_TYPE_DXSCALE)) {
        return FALSE;
    }
    // Inference Core Elements
    if (!gst_element_register(plugin, "dxpostprocess", GST_RANK_NONE,
                              GST_TYPE_DXPOSTPROCESS)) {
        return FALSE;
    }
    if (!gst_element_register(plugin, "dxinfer", GST_RANK_NONE,
                              GST_TYPE_DXINFER)) {
        return FALSE;
    }
    if (!gst_element_register(plugin, "dxpreprocess", GST_RANK_NONE,
                              GST_TYPE_DXPREPROCESS)) {
        return FALSE;
    }
    return TRUE;
}

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, dxstream,
                  "DX Stream plugin", plugin_init, PACKAGE_VERSION, GST_LICENSE,
                  GST_PACKAGE_NAME, GST_PACKAGE_ORIGIN)
