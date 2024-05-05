def pass_content_docs(docs):
    result = "\n\n".join(doc.page_content for doc in docs)
    return result

