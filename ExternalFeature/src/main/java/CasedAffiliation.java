import org.apache.commons.lang3.StringUtils;
import org.dom4j.Document;
import org.dom4j.Element;
import org.dom4j.io.SAXReader;
import org.dom4j.tree.DefaultElement;
import org.xml.sax.SAXException;

import java.io.*;
import java.util.*;

public class CasedAffiliation {
    private static SAXReader reader = new SAXReader();

    static {
        reader.setValidation(false);
        try {
            reader.setFeature("http://apache.org/xml/features/nonvalidating/load-external-dtd", false);
        } catch (SAXException e) {
            e.printStackTrace();
        }
    }

    private static Map<String, String> extractAuthorList(Element e) {
        Element mcEle = e.element("MedlineCitation");
        String pmid = mcEle.element("PMID").getTextTrim();
        Element atcEle = mcEle.element("Article");
        Element alEle = atcEle.element("AuthorList");
        if (alEle == null) {
            return null;
        }

        Map<String, String> authorAffiMap = new HashMap<String, String>();
        List authorList = alEle.elements("Author");
        if (authorList.size() > 100) {
            System.out.println(pmid + "\t" + authorList.size());
        }
        if (authorList != null) {
            for (int i = 0; i < authorList.size(); i++) {
                Object o = authorList.get(i);
                if (o instanceof DefaultElement) {
                    Element authorEle = (DefaultElement) o;
                    Element affinfoEle = authorEle.element("AffiliationInfo");
                    String aff = "";
                    if (affinfoEle != null) {
                        aff = affinfoEle.elementTextTrim("Affiliation").trim();
                        if (aff.length() > 0) {
                            String pm_ao = pmid + "_" + (i + 1);
                            authorAffiMap.put(pm_ao, aff);
                        }
                    }
                }
            }
        }
        return authorAffiMap;

    }

    public static void main(String[] args) throws Exception {
        HashMap<String, String> map0 = parseDatasetAffiliation(new File("/home/zhangli/mydisk-2t/repo/pubmed-paper-author-link/resources/song-dataset-articles"));
        System.out.println(map0.size());
//        HashMap<String, String> allAuthorAffiliationMap = parseDatasetAffiliation(new File("/home/zhangli/mydisk-2t/repo/AuthorNameDisambiguation/dataset/articles"));
        HashMap<String, String> map1 = parseDatasetAffiliation(new File("/home/zhangli/mydisk-2t/repo/pubmed-paper-author-link/resources/gs-dataset-articles"));
        System.out.println(map1.size());
        HashMap<String, String> merge = new HashMap<String, String>();
        merge.putAll(map0);
        merge.putAll(map1);
        System.out.println(merge.size() + "->" + new HashSet<String>(merge.values()).size());
        BufferedWriter bw = new BufferedWriter(new FileWriter(new File("/home/zhangli/mydisk-2t/repo/pubmed-paper-author-link/resources/ner_cased_affiliation.tsv")));
        for (String pmId : merge.keySet()) {
            String text = merge.get(pmId);
            Object[] objects = NerUtil.parseOrganizationLocation(text);
            List<String> orgList = (List<String>) objects[0];
            List<String> locList = (List<String>) objects[1];

            String orgListStr = StringUtils.join(orgList, "|");

            String locListStr = StringUtils.join(locList, "|");
            String[] strings = {pmId, text, orgListStr, locListStr};
//            bw.write(pmId + "\t" + text + "\t" + (orgListStr != null ? orgListStr : "null") + "\t" + (locListStr != null ? locListStr : "null") + "\n");
            bw.write(StringUtils.join(strings, "\t") + "\n");

        }
        bw.close();
    }

    private static HashMap<String, String> parseDatasetAffiliation(File rootFile) {
        File[] files = rootFile.listFiles();
        HashMap<String, String> allAuthorAffiliationMap = new HashMap<String, String>();

        for (File file : files) {
            String name = file.getName();
            if (file.isFile() && file.canRead() && (name.endsWith("xml") || name.endsWith("xml.gz"))) {
                String path = file.getAbsolutePath();
                String fileName = path.substring(path.lastIndexOf(System.getProperty("file.separator")) + 1);

                Document document = null;
                try {
                    BufferedReader br = null;
                    FileInputStream ins = new FileInputStream(path);
                    br = new BufferedReader(new InputStreamReader(ins));
                    document = reader.read(br);
                    br.close();
                } catch (Exception e) {
                    e.printStackTrace();
                    // if there is any error, exit this thread
                    continue;
                }

                Element root = document.getRootElement();
                Iterator eleIter = root.elementIterator();
                while (eleIter.hasNext()) {
                    Object o = eleIter.next();
                    if (o instanceof DefaultElement) {
                        DefaultElement e = (DefaultElement) o;
                        String elename = e.getName();
                        try {
                            if (elename != null && elename.equalsIgnoreCase("PubmedArticle")) {
                                Map<String, String> map = extractAuthorList(e);
                                allAuthorAffiliationMap.putAll(map);
                            } else {
                                System.out.println("NotPubmedArticle, ElementName: " + elename + "\t" + "FileName: " + fileName);
                            }
                        } catch (Exception ee) {
                            ee.printStackTrace();
                            continue;
                        }
                    }
                }
            }
        }
        return allAuthorAffiliationMap;
    }
}
