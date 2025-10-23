import streamlit as st
import folium
from pyproj import Transformer
import json
import numpy as np
from branca.colormap import linear
from opencage.geocoder import OpenCageGeocode
from geopy.distance import geodesic
import math
import requests
import time
from folium.elements import Element
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch


def calculate_radius(height):
    """Calculate the analysis radius based on building height"""
    if height <= 40:
        return max(50 * height, 500)
    elif 40 < height <= 80:
        return 75 * height - 1000
    else:
        return 5000


def fetch_buildings_from_api(center_lat, center_lon, radius):
    """Fetch buildings from 3DBAG API within a bounding box"""
    transformer_to_rd = Transformer.from_crs("EPSG:4326", "EPSG:7415", always_xy=True)
    center_x, center_y, _ = transformer_to_rd.transform(center_lon, center_lat, 0)
    
    bbox_xmin = center_x - radius
    bbox_ymin = center_y - radius
    bbox_xmax = center_x + radius
    bbox_ymax = center_y + radius
    
    base_url = "https://api.3dbag.nl/collections/pand/items"
    all_features = []
    limit = 100
    page = 0
    
    st.info(f"Ophalen van gebouwen uit 3DBAG API...")
    current_url = f"{base_url}?bbox={bbox_xmin},{bbox_ymin},{bbox_xmax},{bbox_ymax}&limit={limit}"
    
    while current_url:
        try:
            response = requests.get(current_url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if 'features' in data:
                features = data['features']
                if features:
                    all_features.extend(features)
                    page += 1
                    st.write(f"Pagina {page} opgehaald: {len(all_features)} gebouwen totaal...")
            
            if 'links' in data:
                next_link = next((link['href'] for link in data['links'] if link['rel'] == 'next'), None)
                current_url = next_link
            else:
                break
                
        except requests.exceptions.RequestException as e:
            st.error(f"Fout bij ophalen van 3DBAG API: {str(e)}")
            break
    
    st.success(f"Succesvol {len(all_features)} gebouwen opgehaald van 3DBAG API")
    return all_features


def process_cityjson_feature(cityjson_feature, center_lat, center_lon, radius):
    """Process a single CityJSONFeature - handles the CityJSON format from 3DBAG API"""
    transformer = Transformer.from_crs("EPSG:7415", "EPSG:4326", always_xy=True)
    
    # CityJSONFeature has: CityObjects, vertices, transform
    if 'CityObjects' not in cityjson_feature or 'vertices' not in cityjson_feature:
        return None
    
    # Get the vertex array and transform (for scaling coordinates)
    vertices_array = cityjson_feature['vertices']
    transform = cityjson_feature.get('transform', {})
    # Default scale to 0.001 (millimeters to meters) and translate to 0
    scale = transform.get('scale', [0.001, 0.001, 0.001])
    translate = transform.get('translate', [0, 0, 0])
    
    # Process each building in CityObjects
    results = []
    
    for obj_id, city_object in cityjson_feature['CityObjects'].items():
        if city_object.get('type') != 'Building':
            continue
        
        # Get building attributes
        attrs = city_object.get('attributes', {})
        building_max = attrs.get('b3_h_dak_50p')
        building_maaiveld = attrs.get('b3_h_maaiveld')
        ground_area = attrs.get('b3_opp_grond')
        
        if building_max is None or building_maaiveld is None:
            continue
        
        building_height = building_max - building_maaiveld
        building_height_rounded = round(building_height)
        
        # Get geometry (usually first geometry at LoD 0 or footprint)
        geometries = city_object.get('geometry', [])
        if not geometries:
            continue
        
        # Use first geometry (footprint)
        geometry = geometries[0]
        boundaries = geometry.get('boundaries', [])
        
        if not boundaries:
            continue
        
        # Convert vertex indices to actual coordinates
        all_vertices = []
        
        try:
            # For MultiSurface: boundaries is list of surfaces
            for surface in boundaries:
                for ring in surface:
                    for vertex_index in ring:
                        if vertex_index < len(vertices_array):
                            # Get vertex and apply transform
                            v = vertices_array[vertex_index]
                            x = v[0] * scale[0] + translate[0]
                            y = v[1] * scale[1] + translate[1]
                            z = v[2] * scale[2] + translate[2]
                            
                            # Transform from RD to WGS84
                            lon, lat, height = transformer.transform(x, y, z)
                            all_vertices.append((lat, lon, height))
        except (IndexError, KeyError, TypeError):
            continue
        
        if not all_vertices:
            continue
        
        # Calculate centroid
        latitudes = [v[0] for v in all_vertices]
        longitudes = [v[1] for v in all_vertices]
        centroid = (np.mean(latitudes), np.mean(longitudes))
        
        # Check if within radius
        distance = geodesic((center_lat, center_lon), centroid).meters
        if distance > radius:
            continue
        
        # Use provided ground area or calculate it
        if ground_area is None:
            coords_2d = [(v[0], v[1]) for v in all_vertices]
            ground_area = calculate_area(coords_2d) if len(coords_2d) >= 3 else 0
        
        results.append({
            'height': building_height_rounded,
            'centroid': centroid,
            'vertices': all_vertices,
            'area': ground_area,
            'distance': distance
        })
    
    return results


def calculate_area(vertices):
    """Simple polygon area calculation"""
    if len(vertices) < 3:
        return 0
    area = 0
    for i in range(len(vertices)):
        j = (i + 1) % len(vertices)
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]
    return abs(area) / 2


def generate_combined_html(quadrant_data, histogram_base64, map_html, address, radius, building_height):
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    total_area_per_quadrant = math.pi * (radius**2) / 4

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Terreinruwheids Analyse - {address}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1, h2 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>Bepaling ruwheidslengte volgens EN-1991-1-4 art. 4.3.2</h1>
        <h2>{address}</h2>
        <p>Rapport gegenereerd op {current_date}</p>
        <ul>
            <li>Gebouwhoogte: {building_height} meters</li>
            <li>Berekende straal: {radius} meters</li>
            <li>Sector oppervlak: {total_area_per_quadrant:.0f} m²</li>
        </ul>
        <h2>Gebouwhoogte distributie</h2>
        <img src="data:image/png;base64,{histogram_base64}" style="max-width: 600px;" />
        <h2>Sector analyse</h2>
        <table>
            <tr>
                <th>Sector</th>
                <th>Sector oppervlak (m²)</th>
                <th>Bebouwd oppervlak (m²)</th>
                <th>Bebouwd percentage (%)</th>
                <th>Gemiddelde hoogte (m)</th>
                <th>z0 (m)</th>
                <th>Status</th>
            </tr>
            {"".join(f"<tr><td>{q['name']}</td><td>{q['total_area']}</td><td>{q['area']}</td><td>{q['built_percentage']}</td><td>{q['avg_height']}</td><td>{q['z0']}</td><td><strong>{q['status']}</strong></td></tr>" for q in quadrant_data)}
        </table>
        <h2>Gebouwhoogte kaart</h2>
        {map_html}
    </body>
    </html>
    """
    return html


def generate_pdf_report(quadrant_data, histogram_base64, map_html, address, radius, building_height):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=16, spaceAfter=30)
    heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'], fontSize=14, spaceAfter=20)
    normal_style = ParagraphStyle('CustomNormal', parent=styles['Normal'], fontSize=10, spaceAfter=10)
    
    story = []
    story.append(Paragraph(f"Terreinruwheidsanalyse - {address}", title_style))
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("Analyse Parameters", heading_style))
    story.append(Paragraph(f"Gebouwhoogte: {building_height} meters", normal_style))
    story.append(Paragraph(f"Berekende straal: {radius} meters", normal_style))
    story.append(Paragraph(f"Sector oppervlak: {math.pi * (radius**2) / 4:.0f} m²", normal_style))
    story.append(Spacer(1, 20))
    
    if histogram_base64:
        try:
            img_data = base64.b64decode(histogram_base64)
            img = Image(BytesIO(img_data), width=6*inch, height=4*inch)
            story.append(img)
            story.append(Spacer(1, 20))
        except:
            story.append(Paragraph("Kon histogram niet toevoegen aan PDF", normal_style))
    
    story.append(Paragraph("Sector Analyse", heading_style))
    
    table_data = [['Sector', 'Oppervlak (m²)', 'Bebouwd (%)', 'Gem. Hoogte (m)', 'z0 (m)', 'Status']]
    for q in quadrant_data:
        table_data.append([q['name'], q['area'], q['built_percentage'], q['avg_height'], q['z0'], q['status']])
    
    table = Table(table_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(table)
    doc.build(story)
    
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


# Streamlit App
st.title("Terreinruwheidsanalyse")
st.write("Analyse van gebouwhoogte en ruwheidslengte volgens NEN-EN 1991-1-4 art 4.3.2.")
st.info("ℹ️ Deze app haalt gebouwdata automatisch op van de 3DBAG API")

if "opencage" in st.secrets and "api_key" in st.secrets["opencage"]:
    opencage_key = st.secrets["opencage"]["api_key"]
else:
    st.error("API Key is not configured. Please check your Streamlit Secrets.")
    st.stop()

if "last_request_time" not in st.session_state:
    st.session_state.last_request_time = 0

def can_make_request():
    current_time = time.time()
    cooldown = 10
    if current_time - st.session_state.last_request_time > cooldown:
        st.session_state.last_request_time = current_time
        return True
    else:
        remaining = cooldown - (current_time - st.session_state.last_request_time)
        st.warning(f"Even geduld. Nieuwe aanvraag mogelijk over {remaining:.1f} seconden.")
        return False

address = st.text_input("Voer het te analyseren adres in", "Kerkstraat 1, Amsterdam")
building_height = st.number_input("Gebouwhoogte in meters", min_value=1.0, value=40.0, step=1.0)

radius = calculate_radius(building_height)
st.info(f"Gebaseerd op een gebouwhoogte van {building_height}m, is de te analyseren straal {radius}m.")

if st.button("Start Analyse") and opencage_key and can_make_request():
    with st.spinner("Verwerken van de gebouwdata..."):
        try:
            geocoder = OpenCageGeocode(opencage_key)
            result = geocoder.geocode(address)
            
            if not result:
                st.error("Adres niet gevonden. Probeer een ander adres.")
                st.stop()
            
            location = result[0]['geometry']
            center_lat, center_lon = location['lat'], location['lng']
            
            features = fetch_buildings_from_api(center_lat, center_lon, radius)
            
            if not features:
                st.error("Geen gebouwen gevonden in dit gebied.")
                st.stop()
            
            # Debug: Show structure of first feature
            if features:
                with st.expander("🔍 Debug: Eerste gebouw structuur"):
                    st.json(features[0])
                    # Show what happens with first vertex
                    if 'vertices' in features[0] and len(features[0]['vertices']) > 0:
                        v = features[0]['vertices'][0]
                        transform = features[0].get('transform', {})
                        scale = transform.get('scale', [0.001, 0.001, 0.001])
                        translate = transform.get('translate', [0, 0, 0])
                        x = v[0] * scale[0] + translate[0]
                        y = v[1] * scale[1] + translate[1]
                        z = v[2] * scale[2] + translate[2]
                        st.write(f"Raw vertex: {v}")
                        st.write(f"Scale: {scale}, Translate: {translate}")
                        st.write(f"Transformed to RD: ({x:.2f}, {y:.2f}, {z:.2f})")
                        
                        transformer_test = Transformer.from_crs("EPSG:7415", "EPSG:4326", always_xy=True)
                        lon, lat, height = transformer_test.transform(x, y, z)
                        st.write(f"Transformed to WGS84: ({lat:.6f}, {lon:.6f}, {height:.2f})")
            
            heights = [0, 0, 0, 0]
            counts = [0, 0, 0, 0]
            areas = [0.0, 0.0, 0.0, 0.0]
            height_area_products = [0.0, 0.0, 0.0, 0.0]
            buildings_within_radius = []
            
            progress_bar = st.progress(0)
            processed_buildings = []
            
            for idx, feature in enumerate(features):
                progress_bar.progress((idx + 1) / len(features))
                building_data = process_cityjson_feature(feature, center_lat, center_lon, radius)
                if building_data:
                    processed_buildings.append(building_data)
                    buildings_within_radius.append(building_data['height'])
            
            st.success(f"{len(processed_buildings)} gebouwen verwerkt binnen de straal")
            
            max_height = max(buildings_within_radius) if buildings_within_radius else 30
            
            m = folium.Map(location=[center_lat, center_lon], zoom_start=16, tiles="cartodbpositron")
            colormap = linear.YlOrRd_09.scale(0, max_height)
            
            folium.Circle(
                location=(center_lat, center_lon),
                radius=radius,
                color="black",
                fill=True,
                fill_opacity=0.1,
                popup=f"Straal: {radius}m<br>Gebouwhoogte: {building_height}m"
            ).add_to(m)
            
            for angle in [0, 90, 180, 270]:
                endpoint_lat = center_lat + (radius / 111320) * math.cos(math.radians(angle))
                endpoint_lon = center_lon + (radius / (111320 * math.cos(math.radians(center_lat)))) * math.sin(math.radians(angle))
                folium.PolyLine(
                    locations=[[center_lat, center_lon], [endpoint_lat, endpoint_lon]],
                    color="black",
                    weight=2,
                    dash_array="5,5"
                ).add_to(m)
            
            for building in processed_buildings:
                centroid = building['centroid']
                height = building['height']
                area = building['area']
                
                if centroid[0] >= center_lat and centroid[1] >= center_lon:
                    quadrant = 0
                elif centroid[0] >= center_lat and centroid[1] < center_lon:
                    quadrant = 1
                elif centroid[0] < center_lat and centroid[1] >= center_lon:
                    quadrant = 2
                else:
                    quadrant = 3
                
                heights[quadrant] += height
                counts[quadrant] += 1
                areas[quadrant] += area
                height_area_products[quadrant] += area * height
                
                color = colormap(height)
                vertices_2d = [(v[0], v[1]) for v in building['vertices']]
                polygon = folium.Polygon(
                    locations=vertices_2d,
                    color=color,
                    fill=True,
                    fill_opacity=0.8,
                    weight=2
                )
                tooltip_text = f"Gebouwhoogte: {height} meters"
                polygon.add_child(folium.Tooltip(tooltip_text))
                polygon.add_to(m)
            
            fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
            counts_hist, bins, _ = plt.hist(buildings_within_radius, bins=10)
            plt.clf()
            
            fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
            patches = ax.bar(bins[:-1], counts_hist, width=np.diff(bins), edgecolor='black', align='edge')
            
            for patch, bin_value in zip(patches, bins[:-1]):
                patch.set_facecolor(colormap(bin_value))
            
            ax.set_xlabel("Hoogte (m)")
            ax.set_ylabel("Aantal")
            plt.title("Distributie gebouwhoogte")
            plt.tight_layout()
            
            buf = BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            histogram_base64 = base64.b64encode(buf.read()).decode('utf-8')
            
            total_land_area_per_quadrant = math.pi * (radius**2) / 4
            z0_values = []
            quadrant_names = ['Noord Oost', 'Noord West', 'Zuid Oost', 'Zuid West']
            quadrant_data = []
            
            stats_html = f"""
            <div style="position: fixed; bottom: 10px; left: 10px; width: 320px; background-color: white;
                        padding: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.3); z-index: 1000;
                        overflow-y: auto; border-radius: 5px;">
            <h4 style="margin-top: 0;">Analyse Parameters</h4>
            <div style="margin-bottom: 8px;">
                <strong>Gebouwhoogte:</strong> {building_height} m<br>
                <strong>Afstand R:</strong> {radius} m<br>
                <strong>Sector oppervlak:</strong> {total_land_area_per_quadrant:.0f} m²
            </div>
            <h4 style="margin-top: 10px;">Gebouwhoogte distributie</h4>
            <img src="data:image/png;base64,{histogram_base64}" width="100%" />
            <h4>Quadrant Analyse</h4>
            """
            
            for i, label in enumerate(quadrant_names):
                avg_height = height_area_products[i] / areas[i] if areas[i] > 0 else 0
                alpha = areas[i] / total_land_area_per_quadrant if total_land_area_per_quadrant > 0 else 0
                alpha_percentage = alpha * 100
                z0 = 0.5 * alpha * avg_height
                z0_values.append(z0)
                
                classification = "bebouwd" if z0 >= 0.5 else "onbebouwd"
                
                quadrant_data.append({
                    "name": label,
                    "area": f"{areas[i]:.0f}",
                    "total_area": f"{total_land_area_per_quadrant:.0f}",
                    "built_percentage": f"{alpha_percentage:.1f}%",
                    "avg_height": f"{avg_height:.1f}",
                    "z0": f"{z0:.2f}",
                    "status": classification
                })
                
                stats_html += f"""
                <div style="margin-bottom: 8px; padding: 5px; background-color: {'#f0f0f0' if i % 2 == 0 else 'white'}; border-radius: 3px;">
                    <strong>{label}</strong><br>
                    Sector oppervlak: {total_land_area_per_quadrant:.0f} m²<br>
                    Bebouwd oppervlak: {areas[i]:.0f} m² ({alpha_percentage:.1f}%)<br>
                    Gemiddelde hoogte: {avg_height:.1f} m<br>
                    z0: {z0:.2f} m<br>
                    Status: <strong>{classification}</strong>
                </div>
                """
            
            stats_html += "</div>"
            folium.Element(stats_html).add_to(m)
            
            quadrant_offsets = [
                (radius / 2 / 111320, radius / 2 / (111320 * math.cos(math.radians(center_lat)))),
                (radius / 2 / 111320, -radius / 2 / (111320 * math.cos(math.radians(center_lat)))),
                (-radius / 2 / 111320, radius / 2 / (111320 * math.cos(math.radians(center_lat)))),
                (-radius / 2 / 111320, -radius / 2 / (111320 * math.cos(math.radians(center_lat))))
            ]
            
            for i, (offset_lat, offset_lon) in enumerate(quadrant_offsets):
                classification = "B" if z0_values[i] >= 0.5 else "O"
                quadrant_center_lat = center_lat + offset_lat
                quadrant_center_lon = center_lon + offset_lon
                
                folium.map.Marker(
                    location=(quadrant_center_lat, quadrant_center_lon),
                    icon=folium.DivIcon(html=f'<div style="font-size: 14pt; font-weight: bold; color: black;">{classification}</div>')
                ).add_to(m)
            
            colormap.add_to(m)
            
            st.subheader("Analyse parameters")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Gebouwhoogte", f"{building_height} m")
            with col2:
                st.metric("Afstand R", f"{radius} m")
            with col3:
                st.metric("Sector oppervlak", f"{total_land_area_per_quadrant:.0f} m²")
            
            st.subheader("Sector Resultaat")
            left_col, right_col = st.columns(2)
            
            for i, label in enumerate(quadrant_names):
                avg_height = height_area_products[i] / areas[i] if areas[i] > 0 else 0
                alpha = areas[i] / total_land_area_per_quadrant
                alpha_percentage = alpha * 100
                z0 = 0.5 * alpha * avg_height
                classification = "bebouwd" if z0 >= 0.5 else "onbebouwd"
                
                col = left_col if i < 2 else right_col
                with col:
                    st.markdown(f"**{label}**")
                    st.metric("Bebouwd oppervlak", f"{areas[i]:.0f} m² ({alpha_percentage:.1f}%)")
                    st.metric("Gemiddelde hoogte", f"{avg_height:.1f} m")
                    st.metric("z0", f"{z0:.2f} m")
                    st.info(f"Status: **{classification}**")
            
            st.subheader("Gebouwhoogte Kaart")
            folium_static(m, width=700, height=700)
            
            map_html = m._repr_html_()
            col1, col2, col3 = st.columns(3)
            
            with col1:
                html_map = m.get_root().render()
                st.download_button(
                    label="Download Kaart (HTML)",
                    data=html_map,
                    file_name="wind_location_map.html",
                    mime="text/html"
                )
            
            with col2:
                combined_html = generate_combined_html(quadrant_data, histogram_base64, map_html, address, radius, building_height)
                st.download_button(
                    label="Download Rapport (HTML)",
                    data=combined_html,
                    file_name="wind_location_analyse.html",
                    mime="text/html"
                )
            
            with col3:
                try:
                    pdf_bytes = generate_pdf_report(quadrant_data, histogram_base64, map_html, address, radius, building_height)
                    st.download_button(
                        label="Download Rapport (PDF)",
                        data=pdf_bytes,
                        file_name="wind_location_analyse.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.warning("PDF export niet beschikbaar")
        
        except Exception as e:
            st.error(f"Er is een fout opgetreden: {str(e)}")
            raise e

st.sidebar.markdown("""
## Instructies
1. Voer het te analyseren adres (+plaats) in
2. Bepaal de gebouwhoogte (m)
3. Klik "Start analyse" om gebouwdata op te halen van 3DBAG API
4. Gebruik "Download Rapport" om een overzicht te downloaden

### About
Deze app analyseert de hoogte van gebouwen en de ruwheidsparameters (z0) in de berekende straal rond een locatie.
Elke sector wordt geclassificeerd als "bebouwd" of "onbebouwd" op basis van de z0-waarde.

### Data bron
Gebouwdata wordt opgehaald via de 3DBAG API (api.3dbag.nl)
""")
