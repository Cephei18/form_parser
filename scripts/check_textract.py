#!/usr/bin/env python3
import base64
import sys

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
except Exception as e:
    print("IMPORT_ERROR:", e)
    sys.exit(2)


def mask_ak(ak):
    if not ak:
        return "NONE"
    s = str(ak)
    if len(s) <= 8:
        return s
    return s[:4] + "..." + s[-4:]


def main():
    session = boto3.Session()
    creds = session.get_credentials()
    if creds is None:
        print("AWS_CREDENTIALS: Not found")
    else:
        frozen = creds.get_frozen_credentials()
        print("AWS_CREDENTIALS: Found, access_key_id=", mask_ak(frozen.access_key))

    region = session.region_name or boto3.session.Session().region_name
    print("AWS_REGION:", region)

    try:
        textract = session.client("textract")
    except Exception as e:
        print("ERROR creating Textract client:", repr(e))
        sys.exit(3)

    # 1x1 PNG (transparent) base64
    b64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
    )
    img = base64.b64decode(b64)

    try:
        print("Calling Textract.detect_document_text() with 1x1 PNG...")
        resp = textract.detect_document_text(Document={"Bytes": img})
        print("TEXTRACT_CALL: success")
        blocks = resp.get("Blocks")
        print("Response summary: Blocks count=", len(blocks) if blocks is not None else 0)
        sys.exit(0)
    except ClientError as e:
        print("AWS_CLIENT_ERROR:", e)
        try:
            print("Error response:\n", e.response)
        except Exception:
            pass
        sys.exit(4)
    except NoCredentialsError as e:
        print("NO_CREDENTIALS_ERROR:", e)
        sys.exit(5)
    except Exception as e:
        print("UNEXPECTED_ERROR:", repr(e))
        sys.exit(6)


if __name__ == "__main__":
    main()
