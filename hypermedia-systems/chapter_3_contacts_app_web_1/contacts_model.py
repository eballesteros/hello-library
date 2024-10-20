from typing import Optional


class Contact:
    # mock contacts database
    db: dict[int, "Contact"] = {}

    def __init__(
        self,
        id_: Optional[int] = None,
        first: Optional[str] = None,
        last: Optional[str] = None,
        phone: Optional[str] = None,
        email: Optional[str] = None,
    ):
        self.id = id_
        self.first = first
        self.last = last
        self.phone = phone
        self.email = email

        self.errors = {}

    def __repr__(self):
        return f"({self.first} {self.last} | {self.phone} | {self.email})"
    
    def update(
            self,
            first: Optional[str] = None,
            last: Optional[str] = None,
            phone: Optional[str] = None,
            email: Optional[str] = None
        ):
            self.first = first
            self.last = last
            self.phone = phone
            self.email = email
    

    def validate(self) -> bool:
        def email_id_duplicate(c: Contact) -> bool:
            # if we're in the process of updating, the new email will
            # have already been set. We check that the id is different
            # to avoid that returning True
            return self.id != c.id and self.email == c.email
        
        if not self.email:
            self.errors['email'] = "Email Required"
        elif any(email_id_duplicate(c) for c in Contact.db.values()):
            self.errors['email'] = "Email already exists"
        return len(self.errors) == 0
    
    def save(self) -> bool:
        if not self.validate():
            return False
        
        # add id if its None
        self.id = self.id or len(Contact.db) + 1

        Contact.db[self.id] = self

        return True
    
    def delete(self) -> bool:
        del Contact.db[self.id]
        

    @classmethod
    def search(cls, text) -> list["Contact"]:
        def is_match(attr: str) -> bool:
            return attr is not None and text.lower() in attr.lower()
        
        def contact_match(c: 'Contact') -> bool:
            return any(
                (
                    is_match(c.first),
                    is_match(c.last),
                    is_match(c.email),
                    is_match(c.phone),
                )
            )
        
        return [
            c for c in cls.db.values() if contact_match(c)
        ]

    @classmethod
    def all(cls) -> list["Contact"]:
        return list(cls.db.values())
    
    @classmethod
    def find(cls, id_: int | str) -> Optional['Contact']:
        c = cls.db.get(int(id_))

        # why?
        if c is not None:
            c.errors = {}

        return c


# populate the db
Contact.db[0] = Contact(
    0, "Raquel", "Buezo", "+1-312-838-3057", "raquel.buezo@gmail.com"
)
Contact.db[1] = Contact(
    1, "Raquel", "Sanchez", "+34-123-456-789", "raquel.schamoso@gmail.com"
)
Contact.db[2] = Contact(
    2, "Tiscar", "Rus", "+34-123-456-789", "mock-email@gmail.com"
)
Contact.db[3] = Contact(
    3, "Eduardo", "Ballesteros", "+1-312-838-3053", "mock-email@gmail.com"
)
Contact.db[4] = Contact(
    4, "Alex", "Sanchez", "+34-123-456-789", "mock-email@gmail.com"
)
Contact.db[5] = Contact(
    5, "Sergio", "Vicens", "+34-123-456-789", "mock-email@gmail.com"
)
Contact.db[6] = Contact(
    6, "Carlos", "Terciado", "+34-123-456-789", "mock-email@gmail.com"
)
