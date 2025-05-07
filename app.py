from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__, static_url_path='/static', static_folder='static')  # Explicit static configuration

# Define shirt options with prices
shirts = {
    "shirt1": {"name": "Classic White Shirt", "image": "shirt1.png", "price": "1000/-"},
    "shirt2": {"name": "Blue Casual Shirt", "image": "shirt2.png", "price": "1500/-"},
    "shirt3": {"name": "Green Polo Shirt", "image": "shirt3.png", "price": "2000/-"},
    "shirt4": {"name": "Light Blue Polo Shirt", "image": "shirt4.png", "price": "1800/-"},
    "shirt5": {"name": "Green Striped Shirt", "image": "shirt5.png", "price": "1399/-"},
    "shirt6": {"name": "Classic SkyBlue Shirt", "image": "shirt6.png", "price": "1000/-"},
    "shirt7": {"name": "White Polo Shirt", "image": "shirt7.png", "price":"999"},
    "shirt8": {"name": "Black Polo Shirt", "image": "shirt8.png", "price": "1800/-"},
    "shirt9": {"name": "Black Striped Shirt", "image": "shirt9.png", "price": "1399/-"},
    "shirt10": {"name": "Classic Skyblue Shirt", "image": "shirt10.png", "price": "1000/-"},
    "shirt11": {"name": "Black Printed Shirt", "image": "shirt11.png", "price":"999"},
    "shirt12": {"name": "Classic White Polo Shirt", "image": "shirt12.png", "price": "1000/-"},
    "shirt13": {"name": "Blue Polo Shirt", "image": "shirt13.png", "price":"999"},

}

# Define women's dresses with prices
dresses = {
    "dress1": {"name": "Floral Green Dress", "image": "dress1.png", "price": "2000/-"},
    "dress2": {"name": "Orange Elegant Dress", "image": "dress2.png", "price": "2500/-"},
    "dress3": {"name": "White Party Dress", "image": "dress3.png", "price": "2200/-"},
    "dress4": {"name": "Bohemian Maxi Dress", "image": "dress4.png", "price": "1800/-"},
    "dress5": {"name": "Yellow Lace Dress", "image": "dress5.png", "price": "2300/-"}, 
    "dress6": {"name": "Bohemian Maxi Dress", "image": "dress6.png", "price": "1800/-"},
    "dress7": {"name": "Yellow Lace Dress", "image": "dress7.png", "price": "2300/-"},
    "dress8": {"name": "Floral Green Dress", "image": "dress8.png", "price":"2000/-"},
    "dress9": {"name": "Orange Elegant Dress", "image": "dress9.png","price":"1999/-"},
    "dress10": {"name": "White Party Dress", "image": "dress10.png", "price": "1899/-"},
    "dress11": {"name":"Red Kurta ", "image":"dress11.png", "price":"999/-"},
    "dress12": {"name":"Yellow Saree", "image":"dress12.png", "price":"2999/-"},
    "dress13": {"name":"Red Crop Top", "image":"dress13.png", "price":"999/-"},
    "dress14": {"name":"Blue Saree", "image":"dress14.png", "price":"2999/-"},
    

}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/products')
def products():
    return render_template('products.html', shirts=shirts, dresses=dresses)

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/tryon', methods=['POST'])
def tryon():
    clothing_id = request.form.get('shirt')  # Using 'shirt' for both shirts and dresses
    if clothing_id in shirts or clothing_id in dresses:
        return render_template('tryon.html', shirt_id=clothing_id)
    return redirect(url_for('products'))

if __name__ == '__main__':
    app.run(debug=True, port=5003)  # Runs on port 5000