from dotenv import load_dotenv; load_dotenv()
import os
print('MAPBOX_TOKEN:', os.environ.get('MAPBOX_TOKEN'))
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from shutil import move

import folium
import streamlit as st
from branca.element import MacroElement
from jinja2 import Template
from huggingface_hub import hf_hub_download
from PIL import Image
from streamlit_folium import st_folium

from osm_ai_helper.run_inference import run_inference
from osm_ai_helper.export_osm import convert_polygons


@st.fragment
def show_map():
    class LatLngPopup(MacroElement):
        _template = Template(
            """
                {% macro script(this, kwargs) %}
                    var {{this.get_name()}} = L.popup();
                    function latLngPop(e) {
                        {{this.get_name()}}
                            .setLatLng(e.latlng)
                            .setContent(e.latlng.lat.toFixed(4) + ", " + e.latlng.lng.toFixed(4))
                            .openOn({{this._parent.get_name()}});
                        }
                    {{this._parent.get_name()}}.on('click', latLngPop);
                {% endmacro %}
                """
        )

        def __init__(self):
            super().__init__()
            self._name = "LatLngPopup"

    m = folium.Map(location=[42.8075, -8.1519], zoom_start=8, tiles="OpenStreetMap")
    m.add_child(LatLngPopup())

    st_folium(m, height=400, width=800)


@st.fragment
def inference(lat_lon):
    with st.spinner("Downloading model..."):
        # TODO: Replace with a model trained to detect golf course features (teeboxes, greens, bunkers, fairways, etc.)
        hf_hub_download(
            "mozilla-ai/swimming-pool-detector",  # Replace this with your golf feature detector model
            filename="model.pt",
            repo_type="model",
            local_dir="models",
        )  # NOTE: This is a placeholder. You must train or provide a golf feature detection model for accurate results.
    with st.spinner("Downloading image and Running inference..."):
        output_path, existing, new, missed = run_inference(
            yolo_model_file="models/model.pt",
            output_dir="/tmp/results",
            lat_lon=lat_lon,
            margin=2,  # 2km x 2km area
            save_full_images=False,
            batch_size=64,
        )
    return output_path, existing, new


@st.fragment
def handle_polygon(polygon):
    raw_image = Image.open(polygon.with_suffix(".png"))
    painted_image = Image.open(f"{polygon.parent}/{polygon.stem}_painted.png")

    st.subheader(f"Reviewing: {polygon.name}")

    col1, col2 = st.columns(2)

    with col1:
        st.image(raw_image, caption="Raw Image", use_container_width=True)
    with col2:
        st.image(painted_image, caption="Painted Image", use_container_width=True)

    if st.button("Keep Polygon", key=f"keep_{polygon}"):
        keep_folder = polygon.parent / "keep"
        keep_folder.mkdir(parents=True, exist_ok=True)
        move(polygon, keep_folder / polygon.name)
        st.success(f"Polygon moved to {keep_folder}")
    elif st.button("Discard Polygon", key=f"discard_{polygon.stem}"):
        discard_folder = polygon.parent / "discard"
        discard_folder.mkdir(parents=True, exist_ok=True)
        move(polygon, discard_folder / polygon.name)
        st.warning(f"Polygon moved to {discard_folder}")


@st.fragment
def download_results(output_path):
    st.divider()
    st.header("Export Results")

    st.markdown(
        "The results will be exported in [OsmChange](https://wiki.openstreetmap.org/wiki/OsmChange) format."
        "\nYou can then import the file in [any of the supported editors](https://wiki.openstreetmap.org/wiki/OsmChange#Editors) format."
    )

    lon_lat_polygons = [
        json.loads(result.read_text())
        for result in (output_path / "keep").glob("*.json")
    ]
    # TODO: Update tags to match the detected feature type, e.g., {"golf": "green"}, {"golf": "bunker"}, etc.
    osmchange = convert_polygons(
        lon_lat_polygons=lon_lat_polygons,
        tags={"golf": "feature"},  # Replace "feature" with the specific type if known
    )
    st.download_button(
        label="Download all polygons in `keep`",
        data=ET.tostring(osmchange, "utf-8"),
        file_name="exported_results.osc",
        mime="type/xml",
    )


st.title("OpenStreetMap Golf Course Feature Helper")

st.markdown(
    """
This demo was created with the repo [mozilla-ai/osm-ai-helper](https://github.com/mozilla-ai/osm-ai-helper).

It is configured to help you find golf course features (teeboxes, greens, fairways, sand traps, etc.) in satellite imagery.\
**NOTE:** You must provide or train a model for golf course features for accurate results.\

You can check the [Create Dataset](https://colab.research.google.com/github/mozilla-ai//osm-ai-helper/blob/main/demo/create_dataset.ipyn)
and [Finetune Model](https://colab.research.google.com/github/mozilla-ai//osm-ai-helper/blob/main/demo/finetune_model.ipynb) notebooks to learn how to train your own model.
"""
)

st.divider()

st.subheader("Search for a golf course, then paste its latitude and longitude.")

st.markdown(
    """
The model will try to find **golf course features** (teeboxes, greens, fairways, sand traps, etc.) in a 2km x 2km area centered on the provided coordinates.\

**Note:** The current model is a placeholder. For accurate results, train or provide a model for golf course features.
"""
)

show_map()

lat_lon = st.text_input("Paste the copied (latitude, longitude)")

if st.button("Run Inference") and lat_lon:
    lat, lon = lat_lon.split(",")
    output_path, existing, new = inference(
        lat_lon=(float(lat.strip()), float(lon.strip()))
    )

    st.info(f"Found {len(existing)} golf course features already in OpenStreetMap.")

    if new:
        st.divider()
        st.header("Review `new` swimming pools")
        st.markdown(
            "Every `new` swimming pool will be displayed at the center of the image in `yellow`."
        )
        st.markdown(
            "Golf course features in other colors are those already existing in OpenStreetMap and they just "
            "indicate whether the model has found them (`green`) or missed them (`red`)."
        )
        for new in Path(output_path).glob("*.json"):
            handle_polygon(new)

        download_results(output_path)
    else:
        st.warning("No `new` swimming pools were found. Try a different location.")
