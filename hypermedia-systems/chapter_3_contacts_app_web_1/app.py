from flask import Flask, redirect, request, render_template, flash
from contacts_model import Contact

app = Flask(__name__)
app.secret_key = 'some_secret_key_wiengpbeuwnr'

@app.route("/")
def index():
    return redirect("/contacts")

@app.route("/contacts")
def contacts():
    search = request.args.get("q")
    if search is not None:
        contacts_set = Contact.search(search)
    else:
        contacts_set = Contact.all()
    return render_template("index.jinja", contacts=contacts_set)

@app.route("/contacts/new", methods=['GET'])
def contacts_new_get():
    return render_template("new.jinja", contact=Contact())

@app.route("/contacts/new", methods=['POST'])
def contacts_new_post():
    c = Contact(
        id_=None,
        first=request.form['first_name'],
        last=request.form['last_name'],
        phone=request.form['phone'],
        email=request.form['email'],
    )
    if c.save():
        flash("Created New Contact!")
        return redirect("/contacts")
    return render_template("new.jinja", contact=c)

@app.route("/contacts/<contact_id>")
def contacts_view(contact_id: int | str = 0):
    contact = Contact.find(contact_id)
    return render_template("show.jinja", contact=contact)

@app.route("/contacts/<contact_id>/edit", methods=["GET"])
def contacts_edit_get(contact_id: int | str = 0):
    contact = Contact.find(contact_id)
    return render_template("edit.jinja", contact=contact)

@app.route("/contacts/<contact_id>/edit", methods=["POST"])
def contacts_edit_post(contact_id: int | str = 0):
    c = Contact.find(contact_id)
    # I think we should validate before updating.
    # Here, the update can succeed, then validation fails
    # then the user sees the error in the edit page
    # buc can walk away leaving the incorrect value
    c.update(
        first=request.form['first_name'],
        last=request.form['last_name'],
        phone=request.form['phone'],
        email=request.form['email'],
    )
    if c.save():
        flash("Updated Contact!")
        return redirect(f"/contacts/{contact_id}")
    else:
        return render_template("edit.jinja", contact=c)

@app.route("/contacts/<contact_id>/delete", methods=["POST"])
def contacts_delete(contact_id: int | str = 0):
    contact = Contact.find(contact_id)
    contact.delete()
    flash("Deleted Contact")
    return redirect("/contacts")