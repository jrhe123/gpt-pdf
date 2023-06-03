export const createPineconeIndex = async (
  client,
  indexName,
  vectorDimension,
) => {
  const existingIndexes = await client.listIndexes();
  if (!existingIndexes.includes(indexName)) {
    console.log(`Creating "${indexName}"...`);
    // Create index
    const createClient = await client.createIndex({
      createRequest: {
        name: indexName,
        dimension: vectorDimension,
        metric: 'cosine',
      },
    });
    console.log(`Index created with client:`, createClient);
    // Wait 60 seconds for index initialization
    await new Promise((resolve) => setTimeout(resolve, 60000));
  } else {
    console.log(`"${indexName}" already exists.`);
  }
};
