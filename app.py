from flask import Flask, render_template, request, redirect, url_for, session

app = Flask(__name__, static_url_path='/static', static_folder='static')
app.secret_key = 'your_secret_key'  # Needed for session management

# Define shirt options with prices
shirts = {
    "shirt1": {"name": "Classic White Shirt", "image": "shirt1.png", "price": "1000/-"},
    "shirt2": {"name": "Blue Casual Shirt", "image": "shirt2.png", "price": "1500/-"},
    "shirt3": {"name": "Green Polo Shirt", "image": "shirt3.png", "price": "2000/-"},
    "shirt4": {"name": "Light Blue Polo Shirt", "image": "shirt4.png", "price": "1800/-"},
    "shirt5": {"name": "Green Striped Shirt", "image": "shirt5.png", "price": "1399/-"},
    "shirt6": {"name": "Classic SkyBlue Shirt", "image": "shirt6.png", "price": "1000/-"},
    # "shirt7": {"name": "White Polo Shirt", "image": "shirt7.png", "price":"999"},
    # "shirt8": {"name": "Black Polo Shirt", "image": "shirt8.png", "price": "1800/-"},
    # "shirt9": {"name": "Black Striped Shirt", "image": "shirt9.png", "price": "1399/-"},
    "shirt10": {"name": "Classic Skyblue Shirt", "image": "shirt10.png", "price": "1000/-"},
    # "shirt11": {"name": "Black Printed Shirt", "image": "shirt11.png", "price":"999"},
    # "shirt12": {"name": "Classic White Polo Shirt", "image": "shirt12.png", "price": "1000/-"},
    # "shirt13": {"name": "Blue Polo Shirt", "image": "shirt13.png", "price":"999"},
}

# Define women's dresses with prices
dresses = {
    # "dress1": {"name": "Floral Green Dress", "image": "dress1.png", "price": "2000/-"},
    "dress2": {"name": "Orange Elegant Dress", "image": "dress2.png", "price": "2500/-"},
    "dress3": {"name": "White Party Dress", "image": "dress3.png", "price": "2200/-"},
    # "dress4": {"name": "Bohemian Maxi Dress", "image": "dress4.png", "price": "1800/-"},
    # "dress5": {"name": "Yellow Lace Dress", "image": "dress5.png", "price": "2300/-"}, 
    "dress6": {"name": "Bohemian Maxi Dress", "image": "dress6.png", "price": "1800/-"},
    # "dress7": {"name": "Yellow Lace Dress", "image": "dress7.png", "price": "2300/-"},
    "dress8": {"name": "Floral Green Dress", "image": "dress8.png", "price":"2000/-"},
    # "dress9": {"name": "Orange Elegant Dress", "image": "dress9.png","price":"1999/-"},
    "dress10": {"name": "White Party Dress", "image": "dress10.png", "price": "1899/-"},
    # "dress11": {"name":"Red Kurta ", "image":"dress11.png", "price":"999/-"},
    # "dress12": {"name":"Yellow Saree", "image":"dress12.png", "price":"2999/-"},
    # "dress13": {"name":"Red Crop Top", "image":"dress13.png", "price":"999/-"},
    # "dress14": {"name":"Blue Saree", "image":"dress14.png", "price":"2999/-"},
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

@app.route('/login', methods=['GET'])
def login_get():
    return render_template('login.html', error=None)

@app.route('/login', methods=['POST'])
def login_post():
    email = request.form.get('email')
    password = request.form.get('password')
    # Simple hardcoded validation for demonstration
    if email == 'admin@example.com' and password == 'password':
        session['user'] = email
        return redirect(url_for('home'))
    else:
        error = 'Invalid email or password'
        return render_template('login.html', error=error)

@app.route('/register', methods=['GET'])
def register_get():
    return render_template('register.html', error=None)

@app.route('/register', methods=['POST'])
def register():
    fullname = request.form.get('fullname')
    email = request.form.get('email')
    password = request.form.get('password')
    contact = request.form.get('contact')
    address = request.form.get('address')

    if not fullname or not email or not password:
        error = "Full Name, Email, and Password are required."
        return render_template('register.html', error=error)

    # Simulate user registration logic here (e.g., save to database)
    # For now, just store user info in session for demonstration
    session['user'] = email

    return redirect(url_for('home'))

@app.route('/add_to_cart', methods=['POST'])
def add_to_cart():
    product_id = request.form.get('product_id')
    if product_id not in shirts and product_id not in dresses:
        flash("Invalid product selected.")
        return redirect(url_for('products'))

    product = shirts.get(product_id) or dresses.get(product_id)

    cart = session.get('cart', {})
    if product_id in cart:
        cart[product_id]['quantity'] += 1
    else:
        cart[product_id] = {
            'name': product['name'],
            'price': product['price'],
            'image': product['image'],
            'quantity': 1
        }

    session['cart'] = cart
    flash(f"{product['name']} added to cart!")
    return redirect(url_for('products'))

@app.route('/cart')
def view_cart():
    cart = session.get('cart', {})
    total_price = 0

    # Convert prices like "1000/-" to integers safely
    for item in cart.values():
        price = int(item['price'].replace("/-", "").replace("Rs.", "").strip())
        total_price += price * item['quantity']

    return render_template('cart.html', cart=cart, total=total_price)

if __name__ == '__main__':
    app.run(debug=True, port=5003)
