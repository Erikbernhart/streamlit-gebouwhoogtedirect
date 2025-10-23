import streamlit as st
import requests
import json
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import io
from datetime import datetime
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from folium.elements import Element

# -----------------------------------------------------
# Helper Functions
# -----------------------------------------------------

def calculate_area(polygon):
    """Calculate area of a polygon in m² using the Shoelace formula."""
    x, y = zip(*polygon)
    return 0.5 * abs(sum(x[i] * y[(i+1) % len(polygon)] - x[(i+1) % len(polygon)] * y[i] for i in range(len(polygon))))

def process_cityjson_feature(cityjson_feature, center_lat, center_lon, radius):
    """Extract building data from a CityJSON feature."""
    processed_buildings = []
    if 'CityObjects' not in cityjson_feature or 'vertices' not in cityjson_feature:
        return None

    for obj_id, obj in cityjson_feature['CityObjects'].items():
        if obj['type'] != 'Building':
            continue

        geom = obj.get('geometry', [])
        if not geom:
            continue

        boundaries = geom[0].get('boundaries', [])
        if not boundaries:
            continue

        vertices = [cityjson_feature['vertices'][v] for surf in boundaries for v in surf[0]]
        if not vertices:
            continue

        lats = [v[1] for v in vertices]
        lons = [v[0] for v in vertices]
        centroid = (np.mean(lats), np.mean(lons))
        distance = geodesic((center_lat, center_lon), centroid).meters
        if distance > radius:
            continue

        height = round(max(v[2] for v in vertices) - min(v[2] for v in vertices))
        area = calculate_area(list(zip(lats, lons)))

        processed_buildings.append({
            "height": height,
            "centroid": centroid,
            "vertices": list(zip(lats, lons)),
            "area": area,
            "distance": distance,
        })

    return processed_buildings if processed_buildings else None

def height_to_color(height):
    if height < 5:
        return "#c7e9b4"
    elif height < 15:
        return "#7fcdbb"
    elif height < 30:
        return "#41b6c4"
    elif height < 60:
        return "#2c7fb8"
    else:
        return "#253494"

def get_quadrant(lat, lon, center_lat, center_lon):
    return (
        "NE" if lat > center_lat and lon > center_lon else
        "NW" if lat > center_lat and lon < center_lon else
        "SE" if lat < center_lat and lon > center_lon else
        "SW"
    )

def generate_combined_html(buildings_within_radius, z0, quadrant, avg_height, avg_area):
    return f"""
    <div style='
        position: fixed;
        top: 10px;
        left: 10px;
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.3);
        z-index: 1000;
        font-family: Arial, sans-serif;
    '>
        <h4>📊 Building Analysis Summary</h4>
        <p><b>Number of Buildings:</b> {len(buildings_within_radius)}</p>
        <p><b>Average Height:</b> {avg_height:.1f} m</p>
        <p><b>Average Ground Area:</b> {avg_area:.1f} m²</p>
        <p><b>Roughness Length (z₀):</b> {z0:.2f} m</p>
        <p><b>Dominant Quadrant:</b> {quadrant}</p>
    </div>
    """

def generate_pdf_report(buildings_within_radius, z0, quadrant, avg_height, avg_area):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = [
        Paragraph("📊 Building Analysis Report", styles["Title"]),
        Spacer(1, 12),
        Paragraph(f"Number of Buildings: {len(buildings_within_radius)}", styles["Normal"]),
        Paragraph(f"Average Height: {avg_height:.1f} m", styles["Normal"]),
        Paragraph(f"Average Ground Area: {avg_area:.1f} m²", styles["Normal"]),
        Paragraph(f"Roughness Length (z₀): {z0:.2f} m", styles["Normal"]),
        Paragraph(f"Dominant Quadrant: {quadrant}", styles["Normal"]),
    ]
    doc.build(elements)
    buffer.seek(0)
    return buffer

# -----------------------------------------------------
# Streamlit App
# -----------------------------------------------------

st.set_page_config(layout="wide")
st.title("🏙️ 3DBAG Building Analyzer")
st.markdown("Select a location and radius to analyze buildings within that area.")

# Map and Draw tool
m = folium.Map(location=[52.3676, 4.9041], zoom_start=13)
Draw(export=False, position="topleft").add_to(m)
st_map = st_folium(m, width=700, height=500)

if st_map.get("last_active_drawing"):
    geom = st_map["last_active_drawing"]["geometry"]
    if geom["type"] == "Point":
        center_lat, center_lon = geom["coordinates"][1], geom["coordinates"][0]
        radius = st.slider("Select analysis radius (m)", 50, 500, 200)

        st.info(f"Analyzing area around lat={center_lat:.5f}, lon={center_lon:.5f} within {radius} m")

        # ✅ Proper API call (your original working fetch)
        bbox = [center_lon - 0.005, center_lat - 0.005, center_lon + 0.005, center_lat + 0.005]
        url = f"https://api.3dbag.nl/collections/pand/items?bbox={bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
        resp = requests.get(url)

        if resp.status_code == 200:
            data = resp.json()
            features = data.get("features", [])
            processed_buildings = []
            buildings_within_radius = []

            progress_bar = st.progress(0)
            for idx, feature in enumerate(features):
                progress_bar.progress((idx + 1) / len(features))

                # Get CityJSON URL for each feature
                cityjson_url = next((link["href"] for link in feature.get("links", []) if link["type"] == "application/city+json"), None)
                if not cityjson_url:
                    continue

                cityjson_resp = requests.get(cityjson_url)
                if cityjson_resp.status_code != 200:
                    continue

                cityjson_feature = cityjson_resp.json()
                buildings = process_cityjson_feature(cityjson_feature, center_lat, center_lon, radius)
                if not buildings:
                    continue

                for b in buildings:
                    processed_buildings.append(b)
                    buildings_within_radius.append(b["height"])

            if not processed_buildings:
                st.warning("No buildings found within this radius.")
            else:
                avg_height = np.mean(buildings_within_radius)
                avg_area = np.mean([b["area"] for b in processed_buildings])
                alpha = 0.1
                z0 = 0.5 * alpha * avg_height

                quadrants = [get_quadrant(b["centroid"][0], b["centroid"][1], center_lat, center_lon)
                             for b in processed_buildings]
                quadrant = max(set(quadrants), key=quadrants.count)

                # Histogram
                fig, ax = plt.subplots()
                ax.hist(buildings_within_radius, bins=10)
                ax.set_xlabel("Building height (m)")
                ax.set_ylabel("Count")
                ax.set_title("Building Height Distribution")
                st.pyplot(fig)

                # Add polygons
                for b in processed_buildings:
                    folium.Polygon(
                        locations=b["vertices"],
                        color=height_to_color(b["height"]),
                        weight=1,
                        fill=True,
                        fill_opacity=0.6,
                    ).add_to(m)

                stats_html = generate_combined_html(buildings_within_radius, z0, quadrant, avg_height, avg_area)
                Element(stats_html).add_to(m)
                st_folium(m, width=700, height=500)

                pdf_buf = generate_pdf_report(buildings_within_radius, z0, quadrant, avg_height, avg_area)
                st.download_button(
                    "📄 Download PDF Report",
                    data=pdf_buf,
                    file_name=f"3DBAG_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
        else:
            st.error(f"Failed to fetch data from 3DBAG API (status {resp.status_code}).")
else:
    st.info("Draw a point on the map to start the analysis.")
