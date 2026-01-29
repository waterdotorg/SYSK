#!/usr/bin/env python3
"""
Admin PIN Generator for SYSK RAG Chatbot
Run this to generate a secure PIN hash for your admin interface
"""

import hashlib
import sys

def generate_pin_hash(pin: str) -> str:
    """Generate SHA-256 hash of PIN"""
    return hashlib.sha256(pin.encode()).hexdigest()

def main():
    print("="*60)
    print("SYSK RAG Chatbot - Admin PIN Generator")
    print("="*60)
    print()
    
    if len(sys.argv) > 1:
        # PIN provided as argument
        pin = sys.argv[1]
    else:
        # Interactive mode
        print("Enter your desired admin PIN:")
        print("(Use a memorable but secure PIN)")
        print()
        pin = input("PIN: ").strip()
    
    if not pin:
        print("Error: PIN cannot be empty")
        sys.exit(1)
    
    # Generate hash
    pin_hash = generate_pin_hash(pin)
    
    print()
    print("="*60)
    print("✅ PIN Hash Generated Successfully!")
    print("="*60)
    print()
    print(f"Your PIN: {pin}")
    print(f"Hash:     {pin_hash}")
    print()
    print("="*60)
    print("Next Steps:")
    print("="*60)
    print()
    print("1. Copy the hash above")
    print()
    print("2. Open sysk_rag_chatbot_with_admin.py")
    print()
    print("3. Find line ~49 with ADMIN_PIN_HASH")
    print()
    print("4. Replace the hash with your new hash:")
    print()
    print(f'   ADMIN_PIN_HASH = "{pin_hash}"')
    print()
    print("5. Save the file")
    print()
    print("6. Restart Streamlit if already running")
    print()
    print("="*60)
    print("⚠️  Security Reminder:")
    print("="*60)
    print()
    print("- Keep your PIN private!")
    print("- Don't share the admin URL (?admin=true) publicly")
    print("- Change PIN regularly for production systems")
    print()

if __name__ == "__main__":
    main()
