import edu.stanford.nlp.ie.AbstractSequenceClassifier;
import edu.stanford.nlp.ie.crf.CRFClassifier;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import org.apache.commons.lang3.StringUtils;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class NerUtil {
    private static String serializedClassifier = "/home/zhangli/mydisk-2t/repo/pubmed-paper-author-link/Dependency-Feature/src/main/resources/english.muc.7class.distsim.crf.ser.gz";
    private static AbstractSequenceClassifier<CoreLabel> classifier;
    private static Pattern p = Pattern.compile("[.,\"\\?!:']");

    static {
        try {
            classifier = CRFClassifier.getClassifier(serializedClassifier);
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }
    }

    public static Object[] parseOrganizationLocation(String str) {
        List<String> orgList = new ArrayList<String>();
        List<String> locList = new ArrayList<String>();
        String res = classifier.classifyToString(str, "slashTags", false);
//        System.out.println("---------------------------------------------------------------------------");
//        System.out.println(str);
//        System.out.println(res);
        List<List<CoreLabel>> classify = classifier.classify(str);
        List<String[]> list = new ArrayList<String[]>();
        List<String> tokenList = new ArrayList<String>();
        for (List<CoreLabel> coreLabels : classify) {
            for (CoreLabel coreLabel : coreLabels) {
                String annotation = coreLabel.get(CoreAnnotations.AnswerAnnotation.class);
//                    System.out.println(annotation);
                String token = coreLabel.toString();
                list.add(new String[]{token, annotation});
                tokenList.add(token);
//                    if (annotation.equalsIgnoreCase("LOCATION") || annotation.equalsIgnoreCase("ORGANIZATION")) {
//                    }
            }
        }

        int size = list.size();
        if (size == 0) {
            return new Object[]{null, null};
        }

        List<Integer> starts = new ArrayList<Integer>();
        List<Integer> ends = new ArrayList<Integer>();
        starts.add(0);
        for (int i = 0; i < size - 1; i++) {
            if (!list.get(i)[1].equals(list.get(i + 1)[1])) {
                starts.add(i + 1);
                ends.add(i + 1);
            }
        }
        ends.add(size);
        assert starts.size() == ends.size();
        for (int i = 0; i < starts.size(); i++) {
            List<String> temp1 = tokenList.subList(starts.get(i), ends.get(i));
            String entity = StringUtils.join(temp1, " ");
            if (list.get(starts.get(i))[1].equalsIgnoreCase("LOCATION")) {
//                System.out.println("Loc: " + entity);
                entity = cleanString(entity);
                if (entity.length() > 0) {
                    locList.add(entity);
                }
            } else if (list.get(starts.get(i))[1].equalsIgnoreCase("ORGANIZATION")) {
//                System.out.println("Org: " + entity);
                entity = cleanString(entity);
                if (entity.length() > 0) {
                    orgList.add(entity);
                }
            }
        }

        return new Object[]{orgList, locList};
    }


    private static String cleanString(String str) {
        str = str.toLowerCase();
        Matcher m = p.matcher(str);
        String first = m.replaceAll(""); //把英文标点符号替换成空，即去掉英文标点符号
        p = Pattern.compile(" {2,}");//去除多余空格
        m = p.matcher(first);
        String res = m.replaceAll(" ");
        return res;
    }
}
